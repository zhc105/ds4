/* The server's disk KV store: chains of immutable segments (ds4_chainstore.h).
 *
 * One file, sealed segment or tail alike:
 *
 *     header (128 bytes, below)
 *     blocks   positions [start rounded down to a block, end): a file that
 *              begins inside a block repeats the rows its parent's last,
 *              partial block already holds, and a restore simply reads them
 *              again (the server's segments begin on a block; a test's need
 *              not)
 *     states   state_bytes each: the one at `end`, then a tail's second one
 *     meta     tokens [start, end), the pictures that begin among them, the
 *              history's whole text from its beginning (what the ids are the
 *              digests of: a file says by itself what it stands for, and a
 *              miss can be explained from the files alone), the trailer
 *
 * Everything a lookup or an eviction needs is in the header, so the index
 * is built from the headers alone and held in memory; a file is opened
 * again only to be resumed.  Files are written whole under a temporary name
 * and renamed, so a file that is there is complete. */

#include "ds4_chainstore.h"
#include "rax.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define CHAIN_MAGIC "DS4CHN01"
#define CHAIN_HEADER 128u
#define CHAIN_SEALED 1u
#define CHAIN_TAIL 2u
#define ID_BYTES DS4_CHAINSTORE_ID_BYTES

static const uint8_t g_no_id[ID_BYTES];

static bool id_is_none(const uint8_t *id) { return !id || memcmp(id, g_no_id, ID_BYTES) == 0; }

static void id_hex(const uint8_t id[ID_BYTES], char out[2 * ID_BYTES + 1]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < ID_BYTES; i++) {
        out[i * 2] = hex[id[i] >> 4];
        out[i * 2 + 1] = hex[id[i] & 15];
    }
    out[2 * ID_BYTES] = '\0';
}

static void put64(uint8_t *p, uint64_t v) {
    ds4_kvstore_le_put32(p, (uint32_t)v);
    ds4_kvstore_le_put32(p + 4, (uint32_t)(v >> 32));
}

static uint64_t get64(const uint8_t *p) {
    return (uint64_t)ds4_kvstore_le_get32(p) | ((uint64_t)ds4_kvstore_le_get32(p + 4) << 32);
}

static bool write_u32(FILE *fp, uint32_t v) {
    uint8_t b[4];
    ds4_kvstore_le_put32(b, v);
    return fwrite(b, 1, 4, fp) == 4;
}

static bool read_u32(FILE *fp, uint32_t *v) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return false;
    *v = ds4_kvstore_le_get32(b);
    return true;
}

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static void set_err(char *err, size_t err_len, const char *msg) {
    if (err && err_len) snprintf(err, err_len, "%s", msg);
}

static void chain_logf(const ds4_chainstore *cs, ds4_kvstore_log_type type, const char *fmt, ...) {
    if (!cs || !cs->log) return;
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    cs->log(cs->log_ud, type, msg);
}

/* ---- ids ----------------------------------------------------------------
 * What a file's rows depend on besides the text is the root of every id:
 * another model, quantization or block size never matches, and its files
 * age out. */

static void chain_root(ds4_engine *engine, ds4_session *session, uint8_t root[ID_BYTES]) {
    uint8_t b[8 + 2 + 4];
    memcpy(b, CHAIN_MAGIC, 8);
    b[8] = (uint8_t)ds4_engine_model_id(engine);
    b[9] = (uint8_t)ds4_engine_routed_quant_bits(engine);
    ds4_kvstore_le_put32(b + 10, ds4_session_block_positions(session));
    ds4_kvstore_sha1 c;
    ds4_kvstore_sha1_init(&c);
    ds4_kvstore_sha1_update(&c, b, sizeof(b));
    ds4_kvstore_sha1_final(&c, root);
}

static void chain_id(const uint8_t root[ID_BYTES], const char *text, size_t len, uint8_t id[ID_BYTES]) {
    ds4_kvstore_sha1 c;
    ds4_kvstore_sha1_init(&c);
    ds4_kvstore_sha1_update(&c, root, ID_BYTES);
    ds4_kvstore_sha1_update(&c, text, len);
    ds4_kvstore_sha1_final(&c, id);
}

/* ---- the index ---------------------------------------------------------- */

static int node_find(const ds4_chainstore *cs, const uint8_t *id) {
    if (id_is_none(id)) return -1;
    for (int i = 0; i < cs->len; i++)
        if (!cs->node[i].tail && memcmp(cs->node[i].id, id, ID_BYTES) == 0) return i;
    return -1;
}

static int node_find_path(const ds4_chainstore *cs, const char *path) {
    for (int i = 0; path && i < cs->len; i++)
        if (!strcmp(cs->node[i].path, path)) return i;
    return -1;
}

static void node_add(ds4_chainstore *cs, const ds4_chainstore_node *n) {
    if (cs->len == cs->cap) {
        cs->cap = cs->cap ? cs->cap * 2 : 64;
        cs->node = realloc(cs->node, (size_t)cs->cap * sizeof(cs->node[0]));
        if (!cs->node) abort();
    }
    cs->node[cs->len++] = *n;
}

/* Forget a node (its file removed, or gone already). */
static void node_forget(ds4_chainstore *cs, int i) {
    free(cs->node[i].path);
    cs->node[i] = cs->node[--cs->len];
}

static void node_remove(ds4_chainstore *cs, int i) {
    unlink(cs->node[i].path);
    node_forget(cs, i);
}

/* The index's links, worked out from the parent fields themselves, which are
 * the files' own headers: nothing is kept beside them to fall out of step
 * with the files.  For every node, where its parent is (LINK_ROOT: it begins
 * a chain; LINK_GONE: the parent is not in the index), whether anything
 * hangs from it, and whether its chain reaches back to position 0 through
 * files that meet end to start.  The segments go in a map by id, and every
 * node looks its parent up there. */
enum { LINK_ROOT = -1, LINK_GONE = -2 };

typedef struct {
    int *up;
    bool *has_child;
    bool *whole;
} chain_links;

static void links_free(chain_links *l) {
    free(l->up);
    free(l->has_child);
    free(l->whole);
}

static chain_links links_build(const ds4_chainstore *cs) {
    const int n = cs->len;
    chain_links l = { malloc((size_t)(n + 1) * sizeof(int)), calloc((size_t)n + 1, sizeof(bool)),
                      calloc((size_t)n + 1, sizeof(bool)) };
    rax *by_id = raxNew();
    for (int i = 0; i < n; i++)
        if (!cs->node[i].tail) raxInsert(by_id, cs->node[i].id, ID_BYTES, (void *)(intptr_t)i, NULL);
    for (int i = 0; i < n; i++) {
        void *p = id_is_none(cs->node[i].parent) ? NULL : raxFind(by_id, cs->node[i].parent, ID_BYTES);
        l.up[i] = id_is_none(cs->node[i].parent) ? LINK_ROOT : p == raxNotFound ? LINK_GONE : (int)(intptr_t)p;
        if (l.up[i] >= 0) l.has_child[l.up[i]] = true;
    }
    raxFree(by_id);
    /* whole: 0 unknown, 1 yes, 2 no, 3 on the walk now (a cycle, which no
     * writer makes, counts as broken) */
    uint8_t *state = calloc((size_t)n + 1, 1);
    int *walk = malloc((size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < n; i++) {
        int depth = 0, at = i;
        uint8_t verdict = 0;
        while (!verdict) {
            if (state[at] == 1 || state[at] == 2) { verdict = state[at]; break; }
            if (state[at] == 3) { verdict = 2; break; }
            state[at] = 3;
            walk[depth++] = at;
            const int up = l.up[at];
            if (up == LINK_ROOT) verdict = cs->node[at].start == 0 ? 1 : 2;
            else if (up == LINK_GONE || cs->node[up].end != cs->node[at].start) verdict = 2;
            else at = up;
        }
        while (depth > 0) state[walk[--depth]] = verdict;
    }
    for (int i = 0; i < n; i++) l.whole[i] = state[i] == 1;
    free(walk);
    free(state);
    return l;
}

static bool header_read(const char *path, ds4_chainstore_node *n) {
    FILE *fp = fopen(path, "rb");
    uint8_t h[CHAIN_HEADER];
    const bool ok = fp && fread(h, 1, sizeof(h), fp) == sizeof(h) &&
                    memcmp(h, CHAIN_MAGIC, 8) == 0 && h[11] == 1 &&
                    (h[8] == CHAIN_SEALED || h[8] == CHAIN_TAIL);
    if (fp) fclose(fp);
    if (!ok) return false;
    memset(n, 0, sizeof(*n));
    n->tail = h[8] == CHAIN_TAIL;
    n->start = ds4_kvstore_le_get32(h + 16);
    n->end = ds4_kvstore_le_get32(h + 20);
    n->text_len = ds4_kvstore_le_get32(h + 24);
    memcpy(n->id, h + 32, ID_BYTES);
    memcpy(n->parent, h + 52, ID_BYTES);
    n->sent_end = ds4_kvstore_le_get32(h + 72);
    n->sent_text_len = ds4_kvstore_le_get32(h + 76);
    memcpy(n->sent_id, h + 80, ID_BYTES);
    n->state_bytes = get64(h + 104);
    struct stat st;
    if (stat(path, &st) != 0 || n->end <= n->start) return false;
    n->file_size = (uint64_t)st.st_size;
    n->last_used = (uint64_t)st.st_mtime;
    return true;
}

uint64_t ds4_chainstore_bytes(const ds4_chainstore *cs) {
    uint64_t bytes = 0;
    for (int i = 0; cs && i < cs->len; i++) bytes += cs->node[i].file_size;
    return bytes;
}

bool ds4_chainstore_open(ds4_chainstore *cs, const char *dir, uint64_t budget_mb, int min_tokens,
                         int n_slots, const char *log_name,
                         void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg), void *log_ud) {
    memset(cs, 0, sizeof(*cs));
    if (!dir || !dir[0]) return false;
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return false;
    cs->dir = strdup(dir);
    cs->budget_bytes = budget_mb * 1024u * 1024u;
    cs->min_tokens = min_tokens;
    cs->segment_tokens = DS4_CHAINSTORE_SEGMENT_TOKENS;
    cs->n_pins = n_slots > 0 ? n_slots : 1;
    cs->pin = calloc((size_t)cs->n_pins, sizeof(cs->pin[0]));
    cs->log_name = log_name ? log_name : "ds4";
    cs->log = log;
    cs->log_ud = log_ud;
    cs->enabled = true;

    int old_files = 0;
    DIR *d = opendir(dir);
    for (struct dirent *de; d && (de = readdir(d));) {
        const size_t len = strlen(de->d_name);
        char *path = ds4_kvstore_path_join(dir, de->d_name);
        const bool ours = len > 4 && (!strcmp(de->d_name + len - 4, ".kvs") || !strcmp(de->d_name + len - 4, ".kvt"));
        ds4_chainstore_node n;
        /* Files of the one-file store that kept this directory before (a
         * digest's name, .kv and its .ckpt companion): nothing resumes them
         * any more, and they are in no index the budget could take them by. */
        const bool old = (len == 43 && !strcmp(de->d_name + 40, ".kv")) ||
                         (len == 45 && !strcmp(de->d_name + 40, ".ckpt"));
        if (strstr(de->d_name, ".tmp.") || old) {
            unlink(path);   /* those, and a store that did not finish */
            old_files += old;
        } else if (ours && header_read(path, &n)) {
            n.path = path;
            path = NULL;
            node_add(cs, &n);
        } else if (ours) {
            unlink(path);
        }
        free(path);
    }
    if (d) closedir(d);
    if (old_files) {
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE, "%s: kv chain store removed %d files of the one-file store",
                   cs->log_name, old_files);
    }
    chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE, "%s: kv chain store %s: %d files, %.1f MiB of %.1f MiB",
               cs->log_name, dir, cs->len, (double)ds4_chainstore_bytes(cs) / 1048576.0,
               (double)cs->budget_bytes / 1048576.0);
    return true;
}

void ds4_chainstore_close(ds4_chainstore *cs) {
    if (!cs) return;
    for (int i = 0; i < cs->len; i++) free(cs->node[i].path);
    free(cs->node);
    free(cs->pin);
    free(cs->dir);
    memset(cs, 0, sizeof(*cs));
}

void ds4_chainstore_pin(ds4_chainstore *cs, int slot, const uint8_t *id) {
    if (!cs || !cs->enabled || slot < 0 || slot >= cs->n_pins) return;
    memcpy(cs->pin[slot], id ? id : g_no_id, ID_BYTES);
}

/* The index is what the files say, and every write asks it for room, so it
 * is made to agree with the directory here first: a file gone (removed by
 * hand, lost) is forgotten, and a node whose chain no longer reaches its
 * beginning (an ancestor gone) is removed, as nothing can be resumed
 * through it; its descendants follow, being cut off too.  Neither needs
 * anything to have been kept in step. */
static void chain_sweep(ds4_chainstore *cs) {
    for (int i = 0; i < cs->len;) {
        struct stat st;
        if (stat(cs->node[i].path, &st) == 0) {
            cs->node[i].file_size = (uint64_t)st.st_size;
            i++;
            continue;
        }
        chain_logf(cs, DS4_KVSTORE_LOG_WARNING, "%s: kv chain file gone: %s", cs->log_name, cs->node[i].path);
        node_forget(cs, i);
    }
    chain_links l = links_build(cs);
    /* last to first: a removal moves the index's last entry into its place,
     * one already looked at */
    for (int i = cs->len - 1; i >= 0; i--) {
        if (l.whole[i]) continue;
        chain_logf(cs, DS4_KVSTORE_LOG_WARNING, "%s: kv chain cut off, removed: tokens=%u..%u %s",
                   cs->log_name, cs->node[i].start, cs->node[i].end, cs->node[i].path);
        node_remove(cs, i);
    }
    links_free(&l);
}

/* A leaf goes before anything it hangs from, and a chain a slot is living on
 * stays: its tail is not on disk while the slot holds the conversation, so
 * its last segment looks like a leaf nobody needs.  Which nodes are leaves is
 * asked of the parent fields each time, not kept. */
void ds4_chainstore_evict(ds4_chainstore *cs, uint64_t extra_bytes) {
    if (!cs || !cs->enabled) return;
    chain_sweep(cs);
    if (cs->budget_bytes == 0) return;
    uint64_t bytes = ds4_chainstore_bytes(cs);
    while (bytes + extra_bytes > cs->budget_bytes) {
        chain_links l = links_build(cs);
        int victim = -1;
        for (int i = 0; i < cs->len; i++) {
            const ds4_chainstore_node *n = &cs->node[i];
            if (l.has_child[i]) continue;
            bool pinned = false;
            for (int p = 0; p < cs->n_pins && !n->tail; p++) pinned |= memcmp(cs->pin[p], n->id, ID_BYTES) == 0;
            if (pinned) continue;
            if (victim < 0 || n->last_used < cs->node[victim].last_used) victim = i;
        }
        links_free(&l);
        if (victim < 0) break;
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE, "%s: kv chain evicted %s tokens=%u..%u %.1f MiB",
                   cs->log_name, cs->node[victim].tail ? "tail" : "segment",
                   cs->node[victim].start, cs->node[victim].end,
                   (double)cs->node[victim].file_size / 1048576.0);
        bytes -= cs->node[victim].file_size;
        node_remove(cs, victim);
    }
}

void ds4_chainstore_drop_tail(ds4_chainstore *cs, const char *tail_path) {
    const int i = cs && cs->enabled ? node_find_path(cs, tail_path) : -1;
    if (i >= 0 && cs->node[i].tail) node_remove(cs, i);
}

static void node_touch(ds4_chainstore_node *n) {
    n->last_used = (uint64_t)time(NULL);
    (void)utimes(n->path, NULL);
}

/* ---- writing ------------------------------------------------------------ */

typedef struct {
    size_t engine_index;
    uint32_t position;
    uint32_t text_len;
    uint8_t id[ID_BYTES];
} chain_state;

/* One file of the live history's positions [start, end): its blocks, the
 * states given (the one at `end` first), its meta.  Written under a
 * temporary name and renamed to `path`. */
static bool chain_write(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                        const char *path, uint8_t kind, const uint8_t *parent, uint32_t start,
                        const chain_state *st, uint32_t n_states, const char *text,
                        const ds4_kvstore_trailer_hooks *hooks, uint64_t *size_out,
                        char *err, size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    const uint32_t end = st[0].position;
    const uint32_t bp = ds4_session_block_positions(session);
    const uint64_t state_bytes = ds4_session_state_bytes(session, st[0].engine_index);
    size_t n_images = 0;
    const ds4_vision_identity *images = ds4_session_vision_identities(session, &n_images);

    uint64_t trailer_bytes = 0;
    if (hooks && hooks->serialized_size && !hooks->serialized_size(hooks->ud, text, &trailer_bytes)) {
        set_err(err, err_len, "the trailer cannot be sized");
        return false;
    }
    uint64_t need = CHAIN_HEADER + ds4_session_block_bytes(session, start - start % bp, end) + n_states * state_bytes +
                    (uint64_t)(end - start) * 4u + st[0].text_len + trailer_bytes + 64u;
    if (cs->budget_bytes != 0 && need > cs->budget_bytes) {
        set_err(err, err_len, "the file is larger than the store's budget");
        return false;
    }
    ds4_chainstore_evict(cs, need);
    /* Under a segment that is gone the file could never be resumed.  The
     * session holds every row from the beginning, so the file begins the
     * chain anew instead (and is that much larger). */
    if (!id_is_none(parent) && node_find(cs, parent) < 0) {
        chain_logf(cs, DS4_KVSTORE_LOG_WARNING, "%s: kv chain parent of tokens=%u.. is gone: written from 0",
                   cs->log_name, start);
        parent = NULL;
        start = 0;
    }
    const uint32_t first_block = start - start % bp;

    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    FILE *fp = fopen(tmp, "wb");
    uint8_t h[CHAIN_HEADER] = {0};
    bool ok = fp && fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    for (uint32_t b = first_block; ok && b < end; b += bp) {
        const uint32_t to = b + bp < end ? b + bp : end;
        ok = ds4_session_write_blocks(session, fp, b, to, err, err_len) == 0;
    }
    for (uint32_t i = 0; ok && i < n_states; i++) {
        ok = ds4_session_state_bytes(session, st[i].engine_index) == state_bytes &&
             ds4_session_write_state(session, st[i].engine_index, fp, err, err_len) == 0;
    }
    const off_t meta = ok ? ftello(fp) : -1;
    ok = ok && meta >= 0 && write_u32(fp, end - start);
    for (uint32_t i = start; ok && i < end; i++) ok = write_u32(fp, (uint32_t)tokens->v[i]);
    uint32_t n_own = 0;
    for (size_t i = 0; i < n_images; i++) n_own += images[i].token_start >= start && images[i].token_start < end;
    ok = ok && write_u32(fp, n_own);
    for (size_t i = 0; ok && i < n_images; i++) {
        if (images[i].token_start < start || images[i].token_start >= end) continue;
        ok = write_u32(fp, images[i].token_start) && write_u32(fp, images[i].token_count) &&
             fwrite(images[i].fingerprint, 1, sizeof(images[i].fingerprint), fp) == sizeof(images[i].fingerprint);
    }
    ok = ok && write_u32(fp, st[0].text_len) && fwrite(text, 1, st[0].text_len, fp) == st[0].text_len;
    uint64_t written = 0;
    uint8_t tb[8];
    const off_t trailer_at = ok ? ftello(fp) : -1;
    put64(tb, 0);
    ok = ok && trailer_at >= 0 && fwrite(tb, 1, 8, fp) == 8;
    if (ok && hooks && hooks->write && trailer_bytes != 0) ok = hooks->write(hooks->ud, fp, text, &written);
    const off_t size = ok ? ftello(fp) : -1;
    if (ok && written != 0) {
        put64(tb, written);
        ok = fseeko(fp, trailer_at, SEEK_SET) == 0 && fwrite(tb, 1, 8, fp) == 8;
    }

    memcpy(h, CHAIN_MAGIC, 8);
    h[8] = kind;
    h[9] = (uint8_t)ds4_engine_model_id(engine);
    h[10] = (uint8_t)ds4_engine_routed_quant_bits(engine);
    h[11] = 1;
    ds4_kvstore_le_put32(h + 12, bp);
    ds4_kvstore_le_put32(h + 16, start);
    ds4_kvstore_le_put32(h + 20, end);
    ds4_kvstore_le_put32(h + 24, st[0].text_len);
    ds4_kvstore_le_put32(h + 28, (uint32_t)ds4_session_ctx(session));
    memcpy(h + 32, st[0].id, ID_BYTES);
    memcpy(h + 52, parent ? parent : g_no_id, ID_BYTES);
    if (n_states > 1) {
        ds4_kvstore_le_put32(h + 72, st[1].position);
        ds4_kvstore_le_put32(h + 76, st[1].text_len);
        memcpy(h + 80, st[1].id, ID_BYTES);
    }
    ds4_kvstore_le_put32(h + 100, n_states);
    put64(h + 104, state_bytes);
    put64(h + 112, ok ? (uint64_t)meta : 0);
    put64(h + 120, (uint64_t)time(NULL));
    ok = ok && size >= 0 && fseeko(fp, 0, SEEK_SET) == 0 && fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    if (fp && fclose(fp) != 0) ok = false;
    ok = ok && rename(tmp, path) == 0;
    if (!ok) {
        unlink(tmp);
        if (err && err_len && !err[0]) set_err(err, err_len, strerror(errno));
        return false;
    }
    *size_out = (uint64_t)size;
    return true;
}

/* The live history's text, and how much of it stands for the first `upto`
 * tokens.  NULL when the history stops inside a picture. */
static char *history_text(ds4_engine *engine, ds4_session *session, uint32_t upto, size_t *len) {
    ds4_tokens prefix = *ds4_session_tokens(session);
    prefix.len = (int)upto;
    size_t n_images = 0;
    const ds4_vision_identity *images = ds4_session_vision_identities(session, &n_images);
    while (n_images > 0 && images[n_images - 1].token_start >= upto) n_images--;
    return ds4_kvstore_render_history_text(engine, &prefix, images, n_images, len);
}

bool ds4_chainstore_seal(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                         const uint8_t *parent, uint32_t sealed_end,
                         const ds4_kvstore_trailer_hooks *hooks,
                         uint8_t id_out[ID_BYTES], char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!cs || !cs->enabled || ds4_session_block_positions(session) == 0) return false;
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens || (uint32_t)tokens->len <= sealed_end || ds4_session_state_count(session) == 0 ||
        ds4_session_state_position(session, 0) != (uint32_t)tokens->len) {
        set_err(err, err_len, "the session has no live state past the last segment");
        return false;
    }
    size_t text_len = 0;
    char *text = history_text(engine, session, (uint32_t)tokens->len, &text_len);
    if (!text || text_len > UINT32_MAX) {
        free(text);
        set_err(err, err_len, "the history's text cannot be rendered");
        return false;
    }
    uint8_t root[ID_BYTES];
    chain_root(engine, session, root);
    chain_state st = { 0, (uint32_t)tokens->len, (uint32_t)text_len, {0} };
    chain_id(root, text, text_len, st.id);
    memcpy(id_out, st.id, ID_BYTES);

    const int have = node_find(cs, st.id);
    if (have >= 0) {   /* another conversation with this history sealed it */
        node_touch(&cs->node[have]);
        free(text);
        return true;
    }
    char name[2 * ID_BYTES + 8];
    id_hex(st.id, name);
    strcat(name, ".kvs");
    char *path = ds4_kvstore_path_join(cs->dir, name);
    const double t0 = now_sec();
    uint64_t size = 0;
    const bool ok = chain_write(cs, engine, session, path, CHAIN_SEALED, parent, sealed_end, &st, 1,
                                text, hooks, &size, err, err_len);
    ds4_chainstore_node n;
    if (ok && header_read(path, &n)) {
        n.path = path;
        path = NULL;
        node_add(cs, &n);
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE,
                   "%s: kv chain sealed tokens=%u..%u text=%zu size=%.2f MiB save=%.1f ms file=%s",
                   cs->log_name, sealed_end, (uint32_t)tokens->len, text_len,
                   (double)size / 1048576.0, (now_sec() - t0) * 1000.0, n.path);
    }
    free(path);
    free(text);
    return ok;
}

char *ds4_chainstore_store_tail(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                                const uint8_t *parent, uint32_t sealed_end, int sent_len,
                                const char *tail_path, const char *reason,
                                const ds4_kvstore_trailer_hooks *hooks, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (!cs || !cs->enabled || ds4_session_block_positions(session) == 0) return NULL;
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens || tokens->len < cs->min_tokens || (uint32_t)tokens->len <= sealed_end) return NULL;
    const size_t n_engine = ds4_session_state_count(session);
    if (n_engine == 0 || ds4_session_state_position(session, 0) != (uint32_t)tokens->len) {
        set_err(err, err_len, "the session has no live state to store");
        return NULL;
    }
    size_t text_len = 0;
    char *text = history_text(engine, session, (uint32_t)tokens->len, &text_len);
    if (!text || text_len > UINT32_MAX) {
        free(text);
        set_err(err, err_len, "the history's text cannot be rendered");
        return NULL;
    }
    uint8_t root[ID_BYTES];
    chain_root(engine, session, root);
    chain_state st[2] = { { 0, (uint32_t)tokens->len, (uint32_t)text_len, {0} } };
    chain_id(root, text, text_len, st[0].id);
    uint32_t n_states = 1;
    for (size_t i = 1; i < n_engine && sent_len > (int)sealed_end && sent_len < tokens->len; i++) {
        if (ds4_session_state_position(session, i) != (uint32_t)sent_len) continue;
        size_t sent_text = 0;
        char *prefix = history_text(engine, session, (uint32_t)sent_len, &sent_text);
        /* the text of a prefix is a prefix of the text: tokens render on their own */
        if (prefix && sent_text <= text_len && memcmp(prefix, text, sent_text) == 0) {
            st[1] = (chain_state){ i, (uint32_t)sent_len, (uint32_t)sent_text, {0} };
            chain_id(root, text, sent_text, st[1].id);
            n_states = 2;
        }
        free(prefix);
        break;
    }

    /* The tail this slot had is replaced while it hangs where this one
     * does.  A history that went back behind its segment left that tail on
     * a branch of its own, which stays as it is; the new one gets a name. */
    char *path = NULL;
    const int old = node_find_path(cs, tail_path);
    if (old >= 0 && cs->node[old].tail &&
        memcmp(cs->node[old].parent, parent ? parent : g_no_id, ID_BYTES) == 0) {
        path = strdup(tail_path);
    } else {
        static unsigned serial;
        char name[64];
        snprintf(name, sizeof(name), "tail-%lx-%lx-%x.kvt", (unsigned long)time(NULL),
                 (unsigned long)getpid(), serial++);
        path = ds4_kvstore_path_join(cs->dir, name);
    }
    const double t0 = now_sec();
    uint64_t size = 0;
    bool ok = chain_write(cs, engine, session, path, CHAIN_TAIL, parent, sealed_end, st, n_states,
                          text, hooks, &size, err, err_len);
    ds4_chainstore_node n;
    ok = ok && header_read(path, &n);
    if (ok) {
        const int was = node_find_path(cs, path);
        if (was >= 0) {   /* the file was replaced: so is its entry */
            node_forget(cs, was);
        }
        n.path = strdup(path);
        node_add(cs, &n);
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE,
                   "%s: kv chain tail stored tokens=%u..%d sent=%u reason=%s size=%.2f MiB save=%.1f ms file=%s",
                   cs->log_name, sealed_end, tokens->len, n_states > 1 ? st[1].position : 0u,
                   reason ? reason : "unknown", (double)size / 1048576.0, (now_sec() - t0) * 1000.0, path);
    } else {
        free(path);
        path = NULL;
    }
    free(text);
    return path;
}

/* ---- lookup -------------------------------------------------------------
 * Every state on disk is (text length, digest of that much text under the
 * root).  The request's text is digested once, stopping at each length some
 * state has; the longest state found whose chain is whole is resumed. */

typedef struct {
    uint32_t text_len;
    int node;
    bool sent;      /* a tail's second state */
} chain_key;

static int chain_key_cmp(const void *a, const void *b) {
    const chain_key *x = a, *y = b;
    return x->text_len < y->text_len ? -1 : x->text_len > y->text_len;
}

/* The longest state whose text begins `text`, of a chain that is whole (l). */
static bool chain_find(const ds4_chainstore *cs, const chain_links *l, const uint8_t root[ID_BYTES],
                       const char *text, size_t text_len, chain_key *out) {
    chain_key *keys = malloc(((size_t)cs->len * 2 + 1) * sizeof(*keys));
    int n = 0;
    for (int i = 0; i < cs->len; i++) {
        const ds4_chainstore_node *d = &cs->node[i];
        if (d->text_len <= text_len) keys[n++] = (chain_key){ d->text_len, i, false };
        if (d->tail && d->sent_end != 0 && d->sent_text_len <= text_len) {
            keys[n++] = (chain_key){ d->sent_text_len, i, true };
        }
    }
    qsort(keys, (size_t)n, sizeof(*keys), chain_key_cmp);
    ds4_kvstore_sha1 c;
    ds4_kvstore_sha1_init(&c);
    ds4_kvstore_sha1_update(&c, root, ID_BYTES);
    size_t fed = 0;
    bool found = false;
    uint8_t digest[ID_BYTES];
    for (int k = 0; k < n; k++) {
        if (k == 0 || keys[k].text_len != keys[k - 1].text_len) {
            ds4_kvstore_sha1_update(&c, text + fed, keys[k].text_len - fed);
            fed = keys[k].text_len;
            ds4_kvstore_sha1 at = c;
            ds4_kvstore_sha1_final(&at, digest);
        }
        const ds4_chainstore_node *d = &cs->node[keys[k].node];
        if (memcmp(keys[k].sent ? d->sent_id : d->id, digest, ID_BYTES) != 0) continue;
        if (!l->whole[keys[k].node]) continue;
        *out = keys[k];
        found = true;   /* keys ascend: the last one found is the longest */
    }
    free(keys);
    return found;
}

void ds4_chainstore_ancestor_at(const ds4_chainstore *cs, const uint8_t *id, uint32_t upto,
                                uint8_t id_out[ID_BYTES], uint32_t *end_out) {
    int i = cs && cs->enabled ? node_find(cs, id) : -1;
    chain_links l = i >= 0 ? links_build(cs) : (chain_links){ 0 };
    while (i >= 0 && cs->node[i].end > upto) i = l.up[i];
    memcpy(id_out, i >= 0 ? cs->node[i].id : g_no_id, ID_BYTES);
    *end_out = i >= 0 ? cs->node[i].end : 0;
    links_free(&l);
}

/* ---- loading ------------------------------------------------------------ */

typedef struct {
    ds4_tokens tokens;
    ds4_vision_identity *images;
    size_t n_images;
} chain_history;

/* Read one file of a chain into the session: its blocks up to `upto` and
 * its tokens and pictures onto the history.  fp is left at nothing in
 * particular; the caller seeks to the state it wants. */
static bool chain_read_file(ds4_session *session, FILE *fp, const ds4_chainstore_node *n, uint32_t upto,
                            chain_history *h, uint64_t *states_at, uint64_t *trailer_at,
                            char *err, size_t err_len) {
    uint8_t hd[CHAIN_HEADER];
    if (fread(hd, 1, sizeof(hd), fp) != sizeof(hd)) return false;
    const uint32_t bp = ds4_session_block_positions(session);
    if (ds4_kvstore_le_get32(hd + 12) != bp || ds4_kvstore_le_get32(hd + 28) > (uint32_t)ds4_session_ctx(session) ||
        get64(hd + 104) != ds4_session_state_bytes(session, 0)) {
        set_err(err, err_len, "the file is of another session's shape");
        return false;
    }
    const uint32_t first = n->start - n->start % bp;
    for (uint32_t b = first; b < n->end; b += bp) {
        const uint32_t stored_to = b + bp < n->end ? b + bp : n->end;
        const uint32_t to = stored_to < upto ? stored_to : upto;
        if (to <= b) break;
        if (ds4_session_read_blocks(session, fp, b, to, stored_to, err, err_len) != 0) return false;
    }
    *states_at = CHAIN_HEADER + ds4_session_block_bytes(session, first, n->end);
    const uint64_t meta = get64(hd + 112);
    uint32_t count = 0;
    if (meta > (uint64_t)INT64_MAX || fseeko(fp, (off_t)meta, SEEK_SET) != 0 || !read_u32(fp, &count) ||
        count != n->end - n->start || (uint32_t)h->tokens.len != n->start) {
        set_err(err, err_len, "the file's tokens do not continue the chain");
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t tok = 0;
        if (!read_u32(fp, &tok)) return false;
        ds4_tokens_push(&h->tokens, (int)tok);
    }
    uint32_t n_pictures = 0;
    if (!read_u32(fp, &n_pictures) || n_pictures > 4096) return false;
    h->images = realloc(h->images, (h->n_images + n_pictures + 1) * sizeof(h->images[0]));
    for (uint32_t i = 0; i < n_pictures; i++) {
        ds4_vision_identity *im = &h->images[h->n_images];
        memset(im, 0, sizeof(*im));
        if (!read_u32(fp, &im->token_start) || !read_u32(fp, &im->token_count) ||
            fread(im->fingerprint, 1, sizeof(im->fingerprint), fp) != sizeof(im->fingerprint)) return false;
        h->n_images++;
    }
    uint32_t text_len = 0;
    if (!read_u32(fp, &text_len) || fseeko(fp, (off_t)text_len, SEEK_CUR) != 0) return false;
    const off_t at = ftello(fp);
    if (at < 0) return false;
    *trailer_at = (uint64_t)at;
    return true;
}

int ds4_chainstore_load(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                        const char *prompt_text, const char *const *held_tails,
                        const ds4_kvstore_trailer_hooks *hooks,
                        ds4_chainstore_load_result *result) {
    if (result) memset(result, 0, sizeof(*result));
    if (!cs || !cs->enabled || !prompt_text || ds4_session_block_positions(session) == 0) return 0;
    uint8_t root[ID_BYTES];
    chain_root(engine, session, root);
    chain_key key;
    chain_links l = links_build(cs);
    if (!chain_find(cs, &l, root, prompt_text, strlen(prompt_text), &key)) {
        links_free(&l);
        return 0;
    }

    /* The chain, from the node resumed back to its beginning: copies, marked
     * as in use before anything is read.  Reading blocks takes K/V pages,
     * and a server whose pool has none stores another conversation away
     * from inside that read, on this thread: a store, which may make room
     * and moves the index.  Nothing below looks at the index again. */
    ds4_chainstore_node *chain = malloc((size_t)(cs->len + 1) * sizeof(*chain));
    int depth = 0;
    for (int i = key.node; i >= 0; i = l.up[i]) {
        node_touch(&cs->node[i]);
        chain[depth] = cs->node[i];
        chain[depth++].path = strdup(cs->node[i].path);
    }
    links_free(&l);
    const ds4_chainstore_node *last = &chain[0];
    const uint32_t position = key.sent ? last->sent_end : last->end;

    const double t0 = now_sec();
    char err[160] = {0};
    chain_history h = {0};
    bool ok = true;
    for (int d = depth - 1; ok && d >= 0; d--) {
        const ds4_chainstore_node *n = &chain[d];
        FILE *fp = fopen(n->path, "rb");
        uint64_t states_at = 0, trailer_at = 0;
        ok = fp && chain_read_file(session, fp, n, position, &h, &states_at, &trailer_at, err, sizeof(err));
        if (ok && d == 0) {
            const uint64_t at = states_at + (key.sent ? n->state_bytes : 0);
            ok = at <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)at, SEEK_SET) == 0 &&
                 ds4_session_read_state(session, fp, h.tokens.v, position, h.images, h.n_images,
                                        n->state_bytes, true, err, sizeof(err)) == 0;
            /* a tail resumed whole keeps the state where its client's text
             * ended as one to fall back to, as the slot that wrote it did */
            if (ok && n->tail && !key.sent && n->sent_end != 0) {
                ok = ds4_session_read_state(session, fp, h.tokens.v, n->sent_end, h.images, h.n_images,
                                            n->state_bytes, false, err, sizeof(err)) == 0;
            }
            uint8_t tb[8];
            if (ok && hooks && hooks->load && fseeko(fp, (off_t)trailer_at, SEEK_SET) == 0 &&
                fread(tb, 1, 8, fp) == 8 && get64(tb) != 0) {
                hooks->load(hooks->ud, fp, hooks->load_wanted);
            }
        }
        if (fp) fclose(fp);
    }
    const ds4_tokens *live = ds4_session_tokens(session);
    ok = ok && live && (uint32_t)live->len == position;
    const double load_ms = (now_sec() - t0) * 1000.0;
    if (ok) {
        bool held = false;
        for (int i = 0; held_tails && held_tails[i]; i++) held |= !strcmp(held_tails[i], last->path);
        /* the sealed segment the resumed history ends with: the last of the
         * chain, or the tail's parent */
        const ds4_chainstore_node *sealed = last->tail ? (depth > 1 ? &chain[1] : NULL) : last;
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE,
                   "%s: kv chain hit tokens=%u text=%u files=%d state=%s load=%.1f ms file=%s",
                   cs->log_name, position, key.text_len, depth,
                   !last->tail ? "segment" : key.sent ? "tail-sent" : "tail-live", load_ms, last->path);
        if (result) {
            result->tokens = (int)position;
            result->key_len = key.text_len;
            result->own = last->tail && !held;
            result->tail_path = result->own ? strdup(last->path) : NULL;
            if (sealed) {
                memcpy(result->parent, sealed->id, ID_BYTES);
                result->sealed_end = sealed->end;
            }
            result->segments = depth;
            result->load_ms = load_ms;
        }
    } else {
        chain_logf(cs, DS4_KVSTORE_LOG_KVCACHE, "%s: kv chain load failed at %s: %s load=%.1f ms",
                   cs->log_name, last->path, err[0] ? err : "unreadable", load_ms);
        ds4_session_invalidate(session);
    }
    ds4_tokens_free(&h.tokens);
    free(h.images);
    for (int d = 0; d < depth; d++) free(chain[d].path);
    free(chain);
    return ok ? (int)position : 0;
}

/* The chain a prompt is about to resume is in use: the store made to give up
 * a slot for it must not make room by taking it. */
void ds4_chainstore_touch(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                          const char *prompt_text) {
    if (!cs || !cs->enabled || !prompt_text || ds4_session_block_positions(session) == 0) return;
    uint8_t root[ID_BYTES];
    chain_root(engine, session, root);
    chain_key key;
    chain_links l = links_build(cs);
    if (chain_find(cs, &l, root, prompt_text, strlen(prompt_text), &key))
        for (int i = key.node; i >= 0; i = l.up[i]) node_touch(&cs->node[i]);
    links_free(&l);
}

typedef struct {
    uint64_t last_used;
    int node;
} node_age;

static int node_newer_first(const void *a, const void *b) {
    const uint64_t x = ((const node_age *)a)->last_used, y = ((const node_age *)b)->last_used;
    return x > y ? -1 : x < y;
}

/* Hand the files' trailers to the hook (the server's tool-call memory, which
 * a restart empties and the files still hold), the most recently used first:
 * a conversation's latest file holds every call it made so far.  Until the
 * hook says it has all it wanted. */
void ds4_chainstore_load_trailers(ds4_chainstore *cs, const ds4_kvstore_trailer_hooks *hooks) {
    if (!cs || !cs->enabled || !hooks || !hooks->load) return;
    node_age *order = malloc((size_t)(cs->len + 1) * sizeof(*order));
    for (int i = 0; i < cs->len; i++) order[i] = (node_age){ cs->node[i].last_used, i };
    qsort(order, (size_t)cs->len, sizeof(*order), node_newer_first);
    bool done = false;
    for (int k = 0; k < cs->len && !done; k++) {
        const ds4_chainstore_node *n = &cs->node[order[k].node];
        FILE *fp = fopen(n->path, "rb");
        uint8_t hd[CHAIN_HEADER], tb[8];
        uint32_t count = 0;
        bool ok = fp && fread(hd, 1, sizeof(hd), fp) == sizeof(hd) && get64(hd + 112) <= (uint64_t)INT64_MAX &&
                  fseeko(fp, (off_t)get64(hd + 112), SEEK_SET) == 0 &&
                  read_u32(fp, &count) && fseeko(fp, (off_t)count * 4, SEEK_CUR) == 0 &&
                  read_u32(fp, &count) &&
                  fseeko(fp, (off_t)count * (8 + (off_t)sizeof(((ds4_vision_identity *)0)->fingerprint)), SEEK_CUR) == 0 &&
                  read_u32(fp, &count) && fseeko(fp, (off_t)count, SEEK_CUR) == 0 &&
                  fread(tb, 1, 8, fp) == 8 && get64(tb) != 0;
        if (ok) done = hooks->load(hooks->ud, fp, hooks->load_wanted) < 0;
        if (fp) fclose(fp);
    }
    free(order);
}

void ds4_chainstore_load_result_free(ds4_chainstore_load_result *result) {
    if (!result) return;
    free(result->tail_path);
    memset(result, 0, sizeof(*result));
}
