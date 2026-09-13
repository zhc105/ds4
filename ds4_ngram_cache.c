/* Qwen PLE n-gram row cache (see ds4_ngram_cache.h).
 *
 * Why not the page cache: the table is 48 GiB of hash-addressed 160 B rows,
 * so neighbouring rows are unrelated and a 4 KiB page holds one useful row.
 * Over 1M tokens of real traffic the page cache needed 12.6 GiB for the hit
 * rate a row cache reaches with 512 MiB, and under memory pressure the kernel
 * evicts those pages first.  Rows here are read with O_DIRECT, one aligned
 * sector window per missing row, submitted together through io_uring.
 *
 * Eviction is a single CLOCK over row slots.  Trace simulation put it level
 * with or ahead of LRU and of a hot/cold two-tier design at every budget; a
 * hit only sets a byte.  Slots in use by the pass being served are pinned so
 * the pass cannot evict its own rows.
 *
 * The index is set associative: 8 (row, slot) entries per 64 B bucket, two
 * candidate buckets per row, sized for 40% load.  Under churn at that load a
 * 200M-operation simulation never found both buckets full; at 75% one insert
 * in 115 did.  When it does happen the insert takes over an unpinned entry of
 * the first bucket, slot included.  A dense row -> slot table would cost 4 B
 * for each of the 320M rows, more than a 1 GiB cache itself. */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* O_DIRECT, statx, mlock2 */
#endif

#include "ds4_ngram_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/syscall.h>
#endif

enum {
    NGC_WAYS = 8,            /* entries per index bucket */
    NGC_QD = 64,             /* reads submitted together */
    NGC_HINT_BATCH = 256,    /* prefetch rows between checks for a gather */
    NGC_LOG_SECONDS = 600,
};
enum { NGC_COLD = 0, NGC_HOT = 1, NGC_PINNED = 2 };   /* ref[] states */
#define NGC_NONE UINT32_MAX

typedef struct {
    uint32_t key;    /* row + 1; 0 marks an empty entry, so zero pages start empty */
    uint32_t slot;
} ngc_entry;

typedef struct {
    uint64_t start;
    uint32_t len;
    uint32_t skip;   /* row offset inside the window */
    void    *buf;
} ngc_io;

#ifdef __linux__
typedef struct {
    int fd;
    void *sq_map, *cq_map;
    size_t sq_len, cq_len, sqes_len;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    unsigned *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
} ngc_ring;
#endif

struct ds4_ngram_cache {
    int fd;
    uint64_t data_offset, rows;
    uint32_t row_bytes;
    uint64_t align;          /* direct I/O offset alignment; 1 for buffered reads */

    uint8_t   *data;         /* [n_slots][row_bytes] */
    uint32_t  *slot_row;     /* [n_slots] row held by the slot */
    uint8_t   *ref;          /* [n_slots] CLOCK state */
    ngc_entry *index;        /* [n_buckets][NGC_WAYS] */
    uint8_t   *seen;         /* [rows] bitmap: read at least once */
    uint64_t n_slots, n_used, hand, n_buckets;
    uint64_t data_len, slot_len, ref_len, index_len, seen_len;

    void *bufs[NGC_QD];
#ifdef __linux__
    bool have_ring;
    ngc_ring ring;
#endif

    /* service-thread scratch for one pass */
    uint32_t *pass_slot, *miss;
    uint8_t *pass_miss;      /* row i of the pass was read, not found */
    uint64_t scratch_cap;
    uint32_t hint_batch[NGC_HINT_BATCH];

    pthread_t thread;
    pthread_mutex_t caller;  /* one gather at a time */
    pthread_mutex_t mu;      /* guards everything below */
    pthread_cond_t wake, done;
    bool stop, req_pending, req_ok;
    const uint32_t *req_rows;
    uint64_t req_n;
    uint8_t *req_out;
    uint32_t *hint;
    uint64_t hint_n, hint_pos, hint_cap;
    ds4_ngram_cache_stats pub;   /* published copy of st, plus the caller-side counters */
    time_t last_log;             /* CLOCK_MONOTONIC seconds */
    uint64_t logged_lookups;

    ds4_ngram_cache_stats st;    /* service thread only */
};

static void *ngc_map(uint64_t bytes) {
    void *p = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

/* ---- index ---- */

static inline uint64_t ngc_mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

/* Two buckets from the two halves of one hash, mapped by multiply-shift so
 * the bucket count need not be a power of two and stays proportional to the
 * cache. */
static inline void ngc_buckets(const ds4_ngram_cache *c, uint32_t row, ngc_entry **b1, ngc_entry **b2) {
    const uint64_t h = ngc_mix(row);
    *b1 = c->index + (((h & 0xffffffffull) * c->n_buckets) >> 32) * NGC_WAYS;
    *b2 = c->index + (((h >> 32) * c->n_buckets) >> 32) * NGC_WAYS;
}

static uint32_t ngc_lookup(const ds4_ngram_cache *c, uint32_t row) {
    ngc_entry *b[2];
    ngc_buckets(c, row, &b[0], &b[1]);
    for (int k = 0; k < 2; k++) {
        for (int w = 0; w < NGC_WAYS; w++) {
            if (b[k][w].key == row + 1u) return b[k][w].slot;
        }
    }
    return NGC_NONE;
}

/* Only the entry that still maps row to this slot: a failed or evicted slot
 * may name a row that has since been admitted again elsewhere. */
static void ngc_index_remove(ds4_ngram_cache *c, uint32_t row, uint32_t slot) {
    ngc_entry *b[2];
    ngc_buckets(c, row, &b[0], &b[1]);
    for (int k = 0; k < 2; k++) {
        for (int w = 0; w < NGC_WAYS; w++) {
            if (b[k][w].key == row + 1u && b[k][w].slot == slot) {
                b[k][w].key = 0;
                return;
            }
        }
    }
}

static uint32_t ngc_clock_alloc(ds4_ngram_cache *c) {
    if (c->n_used < c->n_slots) return (uint32_t)c->n_used++;
    /* two sweeps clear every reference bit, so a third finds a victim unless
     * the pass itself pins every slot */
    for (uint64_t step = 0; step < 2u * c->n_slots + 1u; step++) {
        const uint32_t s = (uint32_t)c->hand;
        c->hand = c->hand + 1u == c->n_slots ? 0 : c->hand + 1u;
        if (c->ref[s] == NGC_PINNED) continue;
        if (c->ref[s] == NGC_HOT) {
            c->ref[s] = NGC_COLD;
            continue;
        }
        ngc_index_remove(c, c->slot_row[s], s);
        return s;
    }
    return NGC_NONE;
}

/* Give a missing row an index entry and a slot, evicting as needed.  The
 * caller pins the slot and fills it. */
static uint32_t ngc_admit(ds4_ngram_cache *c, uint32_t row) {
    ngc_entry *b[2];
    ngc_buckets(c, row, &b[0], &b[1]);
    ngc_entry *pos = NULL;
    int best = 0;
    for (int k = 0; k < 2; k++) {
        ngc_entry *free_entry = NULL;
        int n_free = 0;
        for (int w = 0; w < NGC_WAYS; w++) {
            if (b[k][w].key == 0) {
                if (!free_entry) free_entry = &b[k][w];
                n_free++;
            }
        }
        if (n_free > best) {
            best = n_free;
            pos = free_entry;
        }
    }
    uint32_t slot;
    if (pos) {
        slot = ngc_clock_alloc(c);
        if (slot == NGC_NONE) return NGC_NONE;
    } else {
        /* both buckets full: take over an unpinned entry, preferring a cold one */
        for (int k = 0; k < 2 && !pos; k++) {
            for (int w = 0; w < NGC_WAYS; w++) {
                if (c->ref[b[k][w].slot] == NGC_COLD) {
                    pos = &b[k][w];
                    break;
                }
            }
        }
        for (int k = 0; k < 2 && !pos; k++) {
            for (int w = 0; w < NGC_WAYS; w++) {
                if (c->ref[b[k][w].slot] != NGC_PINNED) {
                    pos = &b[k][w];
                    break;
                }
            }
        }
        if (!pos) return NGC_NONE;
        slot = pos->slot;
    }
    pos->key = row + 1u;
    pos->slot = slot;
    c->slot_row[slot] = row;
    return slot;
}

/* ---- reads ---- */

#ifdef __linux__
static void ngc_ring_free(ngc_ring *r) {
    if (r->sqes) munmap(r->sqes, r->sqes_len);
    if (r->cq_map && r->cq_map != r->sq_map) munmap(r->cq_map, r->cq_len);
    if (r->sq_map) munmap(r->sq_map, r->sq_len);
    if (r->fd >= 0) close(r->fd);
    memset(r, 0, sizeof(*r));
    r->fd = -1;
}

static bool ngc_ring_init(ngc_ring *r, unsigned entries) {
    memset(r, 0, sizeof(*r));
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    r->fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (r->fd < 0) return false;
    r->sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    r->cq_len = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single) r->sq_len = r->cq_len = r->sq_len > r->cq_len ? r->sq_len : r->cq_len;
    r->sq_map = mmap(NULL, r->sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQ_RING);
    if (r->sq_map == MAP_FAILED) r->sq_map = NULL;
    r->cq_map = single ? r->sq_map :
        mmap(NULL, r->cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_CQ_RING);
    if (r->cq_map == MAP_FAILED) r->cq_map = NULL;
    r->sqes_len = p.sq_entries * sizeof(struct io_uring_sqe);
    r->sqes = mmap(NULL, r->sqes_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQES);
    if (r->sqes == MAP_FAILED) r->sqes = NULL;
    if (!r->sq_map || !r->cq_map || !r->sqes) {
        ngc_ring_free(r);
        return false;
    }
    const uint8_t *sq = r->sq_map, *cq = r->cq_map;
    r->sq_tail = (unsigned *)(sq + p.sq_off.tail);
    r->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
    r->sq_array = (unsigned *)(sq + p.sq_off.array);
    r->cq_head = (unsigned *)(cq + p.cq_off.head);
    r->cq_tail = (unsigned *)(cq + p.cq_off.tail);
    r->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    r->cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    return true;
}

/* Submit n reads and wait for all of them; res[j] gets read j's result.
 * The ring is ours alone and n never exceeds its size, so the completion
 * queue cannot overflow.  Transient enter errors are retried; anything else
 * means the ring's bookkeeping is wrong, and continuing would pair stale
 * completions with the next pass's buffers. */
static void ngc_ring_run(ngc_ring *r, int fd, const ngc_io *io, unsigned n, int32_t *res) {
    const unsigned tail = *r->sq_tail, mask = *r->sq_mask;
    for (unsigned j = 0; j < n; j++) {
        const unsigned idx = (tail + j) & mask;
        struct io_uring_sqe *sqe = &r->sqes[idx];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = fd;
        sqe->off = io[j].start;
        sqe->addr = (uint64_t)(uintptr_t)io[j].buf;
        sqe->len = io[j].len;
        sqe->user_data = j;
        r->sq_array[idx] = idx;
    }
    *r->sq_tail = tail + n;
    unsigned submitted = 0, reaped = 0;
    while (reaped < n) {
        const int ret = (int)syscall(__NR_io_uring_enter, r->fd, n - submitted, 1u, IORING_ENTER_GETEVENTS, NULL, 0);
        if (ret < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EBUSY) continue;
            fprintf(stderr, "ds4: PLE cache io_uring_enter failed: %s\n", strerror(errno));
            abort();
        }
        submitted += (unsigned)ret;
        unsigned head = *r->cq_head;
        for (; head != *r->cq_tail; head++, reaped++) {
            const struct io_uring_cqe *cqe = &r->cqes[head & *r->cq_mask];
            res[cqe->user_data] = cqe->res;
        }
        *r->cq_head = head;
    }
}
#endif

static bool ngc_pread(int fd, void *buf, uint32_t len, uint64_t off, uint32_t need) {
    uint32_t got = 0;
    while (got < need) {
        const ssize_t r = pread(fd, (char *)buf + got, len - got, (off_t)(off + got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        got += (uint32_t)r;
    }
    return true;
}

/* Read misses [k0, k0 + n) of the pass into their slots.  Direct I/O wants
 * whole sectors, so each read covers the aligned window around the row
 * (two sectors when the row straddles a boundary); the rest is discarded,
 * since hash-addressed neighbours are almost never wanted.  A read the ring
 * could not complete gets one synchronous retry. */
static bool ngc_read_batch(ds4_ngram_cache *c, const uint32_t *rows, uint64_t k0, unsigned n) {
    ngc_io io[NGC_QD];
    int32_t res[NGC_QD];
    const uint32_t rb = c->row_bytes;
    for (unsigned j = 0; j < n; j++) {
        const uint64_t off = c->data_offset + (uint64_t)rows[c->miss[k0 + j]] * rb;
        io[j].start = off / c->align * c->align;
        io[j].len = (uint32_t)((off + rb + c->align - 1u) / c->align * c->align - io[j].start);
        io[j].skip = (uint32_t)(off - io[j].start);
        io[j].buf = c->bufs[j];
        res[j] = -1;
    }
#ifdef __linux__
    if (c->have_ring) ngc_ring_run(&c->ring, c->fd, io, n, res);
#endif
    for (unsigned j = 0; j < n; j++) {
        const uint32_t need = io[j].skip + rb;
        if (res[j] < (int32_t)need && !ngc_pread(c->fd, io[j].buf, io[j].len, io[j].start, need)) {
            fprintf(stderr, "ds4: PLE cache read at offset %llu failed: %s\n",
                    (unsigned long long)io[j].start, errno ? strerror(errno) : "short read");
            return false;
        }
        const uint32_t i = c->miss[k0 + j], row = rows[i];
        memcpy(c->data + (uint64_t)c->pass_slot[i] * rb, (uint8_t *)io[j].buf + io[j].skip, rb);
        if (c->seen[row >> 3] & (1u << (row & 7u))) c->st.rereads++;
        c->seen[row >> 3] |= (uint8_t)(1u << (row & 7u));
    }
    return true;
}

static bool ngc_scratch(ds4_ngram_cache *c, uint64_t n) {
    if (n <= c->scratch_cap) return true;
    uint32_t *a = realloc(c->pass_slot, (size_t)n * sizeof(uint32_t));
    if (a) c->pass_slot = a;
    uint32_t *b = realloc(c->miss, (size_t)n * sizeof(uint32_t));
    if (b) c->miss = b;
    uint8_t *f = realloc(c->pass_miss, (size_t)n);
    if (f) c->pass_miss = f;
    if (!a || !b || !f) return false;
    c->scratch_cap = n;
    return true;
}

/* Make every row of a pass present and pinned, pass_slot[i] naming row i's
 * slot.  *pinned counts the leading rows that were pinned, which the caller
 * unpins whether or not this succeeded.  On failure no row that was admitted
 * without its bytes stays in the index. */
static bool ngc_ensure(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n, uint64_t *pinned, uint64_t *n_read) {
    *pinned = 0;
    *n_read = 0;
    if (!ngc_scratch(c, n)) return false;
    uint64_t n_miss = 0, i = 0;
    bool ok = true;
    for (; i < n; i++) {
        const uint32_t row = rows[i];
        if (row >= c->rows) {
            ok = false;
            break;
        }
        uint32_t s = ngc_lookup(c, row);
        c->pass_miss[i] = s == NGC_NONE;
        if (s == NGC_NONE) {
            s = ngc_admit(c, row);
            if (s == NGC_NONE) {
                fprintf(stderr, "ds4: PLE cache of %llu rows cannot hold one pass\n",
                        (unsigned long long)c->n_slots);
                ok = false;
                break;
            }
            c->miss[n_miss++] = (uint32_t)i;
        }
        c->ref[s] = NGC_PINNED;
        c->pass_slot[i] = s;
    }
    *pinned = i;
    for (uint64_t k = 0; ok && k < n_miss; k += NGC_QD) {
        ok = ngc_read_batch(c, rows, k, (unsigned)(n_miss - k < NGC_QD ? n_miss - k : NGC_QD));
    }
    if (!ok) {
        for (uint64_t k = 0; k < n_miss; k++) ngc_index_remove(c, rows[c->miss[k]], c->pass_slot[c->miss[k]]);
        return false;
    }
    *n_read = n_miss;
    return true;
}

/* A row read by this pass leaves cold and one found in the cache leaves hot,
 * so a row earns CLOCK's second chance only by being asked for again; a
 * third of all rows are never asked for twice.  In pass order, because a
 * repeat later in the pass is such a second ask and must win. */
static void ngc_unpin(ds4_ngram_cache *c, uint64_t pinned) {
    for (uint64_t i = 0; i < pinned; i++) {
        c->ref[c->pass_slot[i]] = c->pass_miss[i] ? NGC_COLD : NGC_HOT;
    }
}

static time_t ngc_now_us(uint64_t *us) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    *us = (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
    return t.tv_sec;
}

static void ngc_log(const ds4_ngram_cache_stats *s, const char *when) {
    fprintf(stderr,
            "ds4: ple_stat%s: %llu lookups, %.1f%% hit, %llu reads + %llu read ahead "
            "(%.1f%% of reads were evicted rows), %llu/%llu slots used, "
            "blocked %.3f s over %llu token passes, %.3f s over %llu prefill chunks\n",
            when, (unsigned long long)s->lookups,
            s->lookups ? 100.0 * (double)(s->lookups - s->reads) / (double)s->lookups : 0.0,
            (unsigned long long)s->reads, (unsigned long long)s->prefetch_reads,
            s->reads + s->prefetch_reads ? 100.0 * (double)s->rereads / (double)(s->reads + s->prefetch_reads) : 0.0,
            (unsigned long long)s->slots_used, (unsigned long long)s->slots,
            (double)s->wait_us[0] / 1e6, (unsigned long long)s->gathers[0],
            (double)s->wait_us[1] / 1e6, (unsigned long long)s->gathers[1]);
}

static bool ngc_serve_gather(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n, uint8_t *out) {
    uint64_t pinned, n_read;
    const bool ok = ngc_ensure(c, rows, n, &pinned, &n_read);
    if (ok) {
        for (uint64_t i = 0; i < n; i++) {
            memcpy(out + i * c->row_bytes, c->data + (uint64_t)c->pass_slot[i] * c->row_bytes, c->row_bytes);
        }
    }
    ngc_unpin(c, pinned);
    c->st.lookups += n;
    c->st.reads += n_read;
    return ok;
}

static void ngc_serve_hint(ds4_ngram_cache *c, uint64_t n) {
    uint64_t pinned, n_read;
    ngc_ensure(c, c->hint_batch, n, &pinned, &n_read);
    ngc_unpin(c, pinned);
    c->st.prefetch_reads += n_read;
}

/* A gather preempts the hint between batches, so it waits for at most one
 * batch of read-ahead. */
static void *ngc_main(void *arg) {
    ds4_ngram_cache *c = arg;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        while (!c->stop && !c->req_pending && c->hint_pos == c->hint_n) pthread_cond_wait(&c->wake, &c->mu);
        if (c->stop) break;
        if (c->req_pending) {
            const uint32_t *rows = c->req_rows;
            const uint64_t n = c->req_n;
            uint8_t *out = c->req_out;
            pthread_mutex_unlock(&c->mu);
            const bool ok = ngc_serve_gather(c, rows, n, out);
            pthread_mutex_lock(&c->mu);
            c->req_ok = ok;
            c->req_pending = false;
            pthread_cond_signal(&c->done);
        } else {
            const uint64_t left = c->hint_n - c->hint_pos;
            const uint64_t k = left < NGC_HINT_BATCH ? left : NGC_HINT_BATCH;
            memcpy(c->hint_batch, c->hint + c->hint_pos, (size_t)k * sizeof(uint32_t));
            c->hint_pos += k;
            pthread_mutex_unlock(&c->mu);
            ngc_serve_hint(c, k);
            pthread_mutex_lock(&c->mu);
        }
        c->st.slots_used = c->n_used;
        const ds4_ngram_cache_stats caller = c->pub;
        c->pub = c->st;
        memcpy(c->pub.gathers, caller.gathers, sizeof caller.gathers);
        memcpy(c->pub.wait_us, caller.wait_us, sizeof caller.wait_us);
    }
    pthread_mutex_unlock(&c->mu);
    return NULL;
}

/* ---- lifetime ---- */

ds4_ngram_cache *ds4_ngram_cache_open(const char *path, uint64_t data_offset, uint64_t rows,
                                      uint32_t row_bytes, uint64_t cache_bytes) {
    if (rows == 0 || rows >= UINT32_MAX || row_bytes == 0) {
        fprintf(stderr, "ds4: PLE cache: unsupported table of %llu rows\n", (unsigned long long)rows);
        return NULL;
    }
    ds4_ngram_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->fd = -1;
#ifdef __linux__
    c->ring.fd = -1;
#endif
    c->data_offset = data_offset;
    c->rows = rows;
    c->row_bytes = row_bytes;
    c->align = 1;

    c->fd = open(path, O_RDONLY);
    struct stat stbuf;
    if (c->fd < 0 || fstat(c->fd, &stbuf) != 0) {
        fprintf(stderr, "ds4: PLE cache cannot open %s: %s\n", path, strerror(errno));
        goto fail;
    }
    if ((uint64_t)stbuf.st_size < data_offset + rows * row_bytes) {
        fprintf(stderr, "ds4: PLE cache: %s is shorter than its table\n", path);
        goto fail;
    }
    uint64_t mem_align = 4096;
#ifdef __linux__
    /* Direct I/O at the alignment the filesystem reports (512 on spark1's
     * NVMe).  Some filesystems take O_DIRECT without reporting one (btrfs);
     * 4096 is a multiple of every sector size.  One aligned read of the
     * first block proves it works; otherwise reads stay buffered, told not
     * to read ahead. */
    uint64_t align = 4096;
#ifdef STATX_DIOALIGN
    struct statx stx;
    if (statx(c->fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) == 0 &&
        (stx.stx_mask & STATX_DIOALIGN) && stx.stx_dio_offset_align) {
        align = stx.stx_dio_offset_align;
        if (stx.stx_dio_mem_align > mem_align) mem_align = stx.stx_dio_mem_align;
    }
#endif
    const int dfd = open(path, O_RDONLY | O_DIRECT);
    void *probe = NULL;
    if (dfd >= 0 && posix_memalign(&probe, (size_t)mem_align, (size_t)align) == 0 &&
        pread(dfd, probe, (size_t)align, 0) > 0) {
        close(c->fd);
        c->fd = dfd;
        c->align = align;
    } else if (dfd >= 0) {
        close(dfd);
    }
    free(probe);
#endif
#ifdef POSIX_FADV_RANDOM
    if (c->align == 1) (void)posix_fadvise(c->fd, 0, 0, POSIX_FADV_RANDOM);
#endif

    c->n_slots = cache_bytes / row_bytes;
    if (c->n_slots > rows) c->n_slots = rows;
    if (c->n_slots == 0) c->n_slots = 1;
    c->n_buckets = (c->n_slots * 5u + 15u) / 16u;   /* n_slots / 3.2: 40% load at 8 ways */
    c->data_len = c->n_slots * row_bytes;
    c->slot_len = c->n_slots * sizeof(uint32_t);
    c->ref_len = c->n_slots;
    c->index_len = c->n_buckets * NGC_WAYS * sizeof(ngc_entry);
    c->seen_len = rows / 8u + 1u;
    c->data = ngc_map(c->data_len);
    c->slot_row = ngc_map(c->slot_len);
    c->ref = ngc_map(c->ref_len);
    c->index = ngc_map(c->index_len);
    c->seen = ngc_map(c->seen_len);
    if (!c->data || !c->slot_row || !c->ref || !c->index || !c->seen) {
        fprintf(stderr, "ds4: PLE cache cannot map %.2f GiB\n",
                (double)(c->data_len + c->slot_len + c->ref_len + c->index_len) / 1073741824.0);
        goto fail;
    }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    (void)madvise(c->data, c->data_len, MADV_HUGEPAGE);
    (void)madvise(c->index, c->index_len, MADV_HUGEPAGE);
#endif
#if defined(__linux__) && defined(MLOCK_ONFAULT)
    /* Swapped-out rows would defeat the point of holding them.  Locked on
     * fault, so memory is committed as the cache fills, not up front. */
    if (mlock2(c->data, c->data_len, MLOCK_ONFAULT) != 0 ||
        mlock2(c->index, c->index_len, MLOCK_ONFAULT) != 0 ||
        mlock2(c->slot_row, c->slot_len, MLOCK_ONFAULT) != 0 ||
        mlock2(c->ref, c->ref_len, MLOCK_ONFAULT) != 0) {
        fprintf(stderr, "ds4: warning: PLE cache not locked in memory (%s); raise RLIMIT_MEMLOCK "
                "(LimitMEMLOCK= for a systemd unit) so it cannot be swapped out\n", strerror(errno));
    }
#endif
    const uint64_t buf_bytes = ((row_bytes + c->align - 1u) / c->align + 1u) * c->align;
    for (int j = 0; j < NGC_QD; j++) {
        if (posix_memalign(&c->bufs[j], (size_t)mem_align, (size_t)buf_bytes) != 0) {
            c->bufs[j] = NULL;
            fprintf(stderr, "ds4: PLE cache cannot allocate read buffers\n");
            goto fail;
        }
    }
#ifdef __linux__
    c->have_ring = ngc_ring_init(&c->ring, NGC_QD);
#endif
    c->st.slots = c->n_slots;
    c->st.bytes = c->data_len + c->slot_len + c->ref_len + c->index_len;
    c->pub = c->st;
    uint64_t start_us;
    c->last_log = ngc_now_us(&start_us);

    pthread_mutex_init(&c->caller, NULL);
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->wake, NULL);
    pthread_cond_init(&c->done, NULL);
    if (pthread_create(&c->thread, NULL, ngc_main, c) != 0) {
        fprintf(stderr, "ds4: PLE cache cannot start its thread\n");
        pthread_cond_destroy(&c->done);
        pthread_cond_destroy(&c->wake);
        pthread_mutex_destroy(&c->mu);
        pthread_mutex_destroy(&c->caller);
        goto fail;
    }
    char mode[32] = "buffered";
    if (c->align > 1) snprintf(mode, sizeof mode, "direct %llu B", (unsigned long long)c->align);
    fprintf(stderr, "ds4: PLE cache: %.2f GiB for %llu rows, %s reads%s\n",
            (double)c->st.bytes / 1073741824.0, (unsigned long long)c->n_slots, mode,
#ifdef __linux__
            c->have_ring ? " via io_uring" : "");
#else
            "");
#endif
    return c;

fail:
    for (int j = 0; j < NGC_QD; j++) free(c->bufs[j]);
    if (c->data) munmap(c->data, c->data_len);
    if (c->slot_row) munmap(c->slot_row, c->slot_len);
    if (c->ref) munmap(c->ref, c->ref_len);
    if (c->index) munmap(c->index, c->index_len);
    if (c->seen) munmap(c->seen, c->seen_len);
    if (c->fd >= 0) close(c->fd);
    free(c);
    return NULL;
}

void ds4_ngram_cache_close(ds4_ngram_cache *c) {
    if (!c) return;
    pthread_mutex_lock(&c->mu);
    c->stop = true;
    pthread_cond_signal(&c->wake);
    pthread_mutex_unlock(&c->mu);
    pthread_join(c->thread, NULL);
    if (c->pub.lookups) ngc_log(&c->pub, " at exit");
    pthread_cond_destroy(&c->done);
    pthread_cond_destroy(&c->wake);
    pthread_mutex_destroy(&c->mu);
    pthread_mutex_destroy(&c->caller);
#ifdef __linux__
    if (c->have_ring) ngc_ring_free(&c->ring);
#endif
    for (int j = 0; j < NGC_QD; j++) free(c->bufs[j]);
    munmap(c->data, c->data_len);
    munmap(c->slot_row, c->slot_len);
    munmap(c->ref, c->ref_len);
    munmap(c->index, c->index_len);
    munmap(c->seen, c->seen_len);
    free(c->pass_slot);
    free(c->miss);
    free(c->pass_miss);
    free(c->hint);
    close(c->fd);
    free(c);
}

bool ds4_ngram_cache_gather(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n, uint8_t *out) {
    if (n == 0) return true;
    uint64_t t0, t1;
    ngc_now_us(&t0);
    pthread_mutex_lock(&c->caller);
    pthread_mutex_lock(&c->mu);
    c->req_rows = rows;
    c->req_n = n;
    c->req_out = out;
    c->req_pending = true;
    pthread_cond_signal(&c->wake);
    while (c->req_pending) pthread_cond_wait(&c->done, &c->mu);
    const bool ok = c->req_ok;
    const time_t now = ngc_now_us(&t1);
    const int large = n > DS4_NGRAM_CACHE_SMALL_PASS;
    c->pub.gathers[large]++;
    c->pub.wait_us[large] += t1 - t0;
    if (now - c->last_log >= NGC_LOG_SECONDS && c->pub.lookups != c->logged_lookups) {
        ngc_log(&c->pub, "");
        c->last_log = now;
        c->logged_lookups = c->pub.lookups;
    }
    pthread_mutex_unlock(&c->mu);
    pthread_mutex_unlock(&c->caller);
    return ok;
}

void ds4_ngram_cache_prefetch(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n) {
    pthread_mutex_lock(&c->mu);
    if (n > c->hint_cap) {
        uint32_t *h = realloc(c->hint, (size_t)n * sizeof(uint32_t));
        if (h) {
            c->hint = h;
            c->hint_cap = n;
        }
    }
    const uint64_t k = n <= c->hint_cap ? n : 0;   /* a hint that cannot be held is dropped */
    if (k) memcpy(c->hint, rows, (size_t)k * sizeof(uint32_t));
    c->hint_n = k;
    c->hint_pos = 0;
    pthread_cond_signal(&c->wake);
    pthread_mutex_unlock(&c->mu);
}

void ds4_ngram_cache_stats_get(ds4_ngram_cache *c, ds4_ngram_cache_stats *out) {
    pthread_mutex_lock(&c->mu);
    *out = c->pub;
    pthread_mutex_unlock(&c->mu);
}
