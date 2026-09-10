#include "ds4_kvstore.h"

/* Shared disk KV checkpoint file support.
 *
 * The low-level file layout and payload helpers are intentionally shared.  The
 * ds4-server still owns the automatic byte-prefix cache policy built on top of
 * this file; ds4-agent uses only the same durable format for explicit sessions,
 * with its own policy in ds4_agent.c.  Protocol-specific extras, such as the
 * server's tool-id -> exact DSML trailer, are attached through trailer hooks and
 * still live with the protocol code that owns those mappings.
 *
 * A file holds one conversation and grows with it:
 *
 *   fixed header (48 bytes)
 *   blocks: what every position of the history contributes, in runs of
 *           block_positions positions, written once and appended to
 *   tail:   "TAIL", the state index, the tokens, the rendered text, the
 *           visible-transcript keys, the state blobs, the trailers
 *
 * A store that continues a file appends the blocks of the new positions and
 * writes a new tail; anything else (a new conversation, an edited history,
 * a fork) is a new file.  Each state in the index is a point the session
 * can resume from, keyed by the prompt bytes it stands for; the first one
 * of a file is kept across stores, so the anchor of a shared system prompt
 * stays available to other conversations. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 2u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * ds4_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 3u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000

#define KV_TAIL_FIXED 32u      /* "TAIL", n_states, blocks_end, block_positions, text_len, tokens, trailer_offset */
#define KV_TAIL_STATE 52u      /* position, key_len, sha, offset, bytes, key_offset */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} kv_buf;

static void kv_die(const char *msg) {
    fprintf(stderr, "ds4-kvstore: %s\n", msg);
    exit(1);
}

static void *kv_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = kv_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void kv_buf_reserve(kv_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) kv_die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = kv_xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void kv_buf_append(kv_buf *b, const void *p, size_t n) {
    kv_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void kv_buf_putc(kv_buf *b, char c) {
    kv_buf_append(b, &c, 1);
}

static void kv_buf_puts(kv_buf *b, const char *s) {
    kv_buf_append(b, s, strlen(s));
}

static void kv_buf_printf(kv_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) kv_die("vsnprintf failed");
    kv_buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *kv_buf_take(kv_buf *b) {
    if (!b->ptr) return kv_xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static double kv_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static const char *kv_log_name(const ds4_kvstore *kc) {
    return kc && kc->log_name ? kc->log_name : "ds4";
}

static void kv_logf(ds4_kvstore *kc, ds4_kvstore_log_type type,
                    const char *fmt, ...) {
    if (!kc || !kc->log) return;
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    char *msg = kv_xmalloc((size_t)n + 1);
    vsnprintf(msg, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    kc->log(kc->log_ud, type, msg);
    free(msg);
}

static void kv_set_err(char *err, size_t err_len, const char *msg) {
    if (err && err_len) snprintf(err, err_len, "%s", msg);
}

ds4_kvstore_options ds4_kvstore_default_options(void) {
    return (ds4_kvstore_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
    };
}

uint8_t ds4_kvstore_reason_code(const char *reason) {
    if (!reason) return DS4_KVSTORE_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return DS4_KVSTORE_REASON_COLD;
    if (!strcmp(reason, "continued")) return DS4_KVSTORE_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return DS4_KVSTORE_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return DS4_KVSTORE_REASON_SHUTDOWN;
    if (!strcmp(reason, "agent-system")) return DS4_KVSTORE_REASON_AGENT_SYSTEM;
    if (!strcmp(reason, "agent-session")) return DS4_KVSTORE_REASON_AGENT_SESSION;
    return DS4_KVSTORE_REASON_UNKNOWN;
}

const char *ds4_kvstore_key_kind(uint8_t ext_flags) {
    if (ext_flags & DS4_KVSTORE_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & DS4_KVSTORE_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

void ds4_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static bool kv_write_u32(FILE *fp, uint32_t v) {
    uint8_t b[4];
    ds4_kvstore_le_put32(b, v);
    return fwrite(b, 1, 4, fp) == 4;
}

static bool kv_read_u32(FILE *fp, uint32_t *v) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return false;
    *v = ds4_kvstore_le_get32(b);
    return true;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

static void sha1_bytes(const void *ptr, size_t len, uint8_t out[20]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    sha1_final(&c, out);
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    uint8_t digest[20];
    sha1_bytes(ptr, len, digest);
    hex20(digest, out);
}

bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

char *ds4_kvstore_path_join(const char *dir, const char *name) {
    kv_buf b = {0};
    kv_buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') kv_buf_putc(&b, '/');
    kv_buf_puts(&b, name);
    return kv_buf_take(&b);
}

char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(kc->dir, name);
}

static bool kv_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = kv_xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

void ds4_kvstore_entry_free(ds4_kvstore_entry *e) {
    free(e->path);
    free(e->state);
    memset(e, 0, sizeof(*e));
}

void ds4_kvstore_clear(ds4_kvstore *kc) {
    for (int i = 0; i < kc->len; i++) ds4_kvstore_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(ds4_kvstore *kc, ds4_kvstore_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = kv_xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

/* =========================================================================
 * File layout
 * ========================================================================= */

static bool kv_write_header(FILE *fp, const ds4_kvstore_entry *e) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    memset(h, 0, sizeof(h));
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = e->quant_bits;
    h[5] = e->reason;
    h[6] = e->ext_flags;
    h[7] = e->model_id;
    ds4_kvstore_le_put32(h + 8, e->tokens);
    ds4_kvstore_le_put32(h + 12, e->hits);
    ds4_kvstore_le_put32(h + 16, e->ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    kv_le_put64(h + 24, e->created_at);
    kv_le_put64(h + 32, e->last_used);
    kv_le_put64(h + 40, e->tail_offset);
    return fseeko(fp, 0, SEEK_SET) == 0 && fwrite(h, 1, sizeof(h), fp) == sizeof(h);
}

static bool kv_read_header(FILE *fp, ds4_kvstore_entry *e) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fseeko(fp, 0, SEEK_SET) != 0 || fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = kv_le_get64(h + 24);
    e->last_used = kv_le_get64(h + 32);
    e->tail_offset = kv_le_get64(h + 40);
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4) &&
           e->tail_offset >= DS4_KVSTORE_FIXED_HEADER;
}

/* The tail's fixed part and state index; the offsets of what follows the
 * index are derived so a reader needs only this much to know the file. */
bool ds4_kvstore_read_index(FILE *fp, ds4_kvstore_entry *e) {
    free(e->state);
    e->state = NULL;
    e->n_states = 0;
    if (!kv_read_header(fp, e)) return false;
    if (e->tail_offset > (uint64_t)INT64_MAX ||
        fseeko(fp, (off_t)e->tail_offset, SEEK_SET) != 0) return false;
    uint8_t t[KV_TAIL_FIXED];
    if (fread(t, 1, sizeof(t), fp) != sizeof(t)) return false;
    if (memcmp(t, "TAIL", 4) != 0) return false;
    e->n_states = ds4_kvstore_le_get32(t + 4);
    e->blocks_end = ds4_kvstore_le_get32(t + 8);
    e->block_positions = ds4_kvstore_le_get32(t + 12);
    e->text_bytes = ds4_kvstore_le_get32(t + 16);
    e->tail_tokens = ds4_kvstore_le_get32(t + 20);
    e->trailer_offset = kv_le_get64(t + 24);
    if (e->n_states > 4096u) return false;
    e->tokens_offset = e->tail_offset + KV_TAIL_FIXED + (uint64_t)e->n_states * KV_TAIL_STATE;
    e->text_offset = e->tokens_offset + (uint64_t)e->tail_tokens * 4u;
    if (e->n_states == 0) return true;
    e->state = kv_xmalloc((size_t)e->n_states * sizeof(e->state[0]));
    for (uint32_t i = 0; i < e->n_states; i++) {
        uint8_t r[KV_TAIL_STATE];
        if (fread(r, 1, sizeof(r), fp) != sizeof(r)) {
            free(e->state);
            e->state = NULL;
            e->n_states = 0;
            return false;
        }
        ds4_kvstore_state *st = &e->state[i];
        st->position = ds4_kvstore_le_get32(r);
        st->key_len = ds4_kvstore_le_get32(r + 4);
        hex20(r + 8, st->sha);
        st->offset = kv_le_get64(r + 28);
        st->bytes = kv_le_get64(r + 36);
        st->key_offset = kv_le_get64(r + 44);
    }
    return true;
}

bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < (off_t)DS4_KVSTORE_FIXED_HEADER) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    bool ok = ds4_kvstore_read_index(fp, &e) &&
              e.text_offset + e.text_bytes <= (uint64_t)st.st_size &&
              e.trailer_offset <= (uint64_t)st.st_size;
    fclose(fp);
    if (!ok) {
        free(e.state);
        return false;
    }
    memcpy(e.sha, sha, 41);
    e.path = kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

char *ds4_kvstore_read_text(FILE *fp, const ds4_kvstore_entry *e) {
    /* the length comes from the file: cap it by what the file holds
     * before allocating */
    if (e->text_offset > (uint64_t)INT64_MAX || fseeko(fp, 0, SEEK_END) != 0) return NULL;
    const off_t end = ftello(fp);
    if (end < 0 || e->text_offset + e->text_bytes > (uint64_t)end ||
        fseeko(fp, (off_t)e->text_offset, SEEK_SET) != 0) return NULL;
    char *text = kv_xmalloc((size_t)e->text_bytes + 1);
    if (fread(text, 1, e->text_bytes, fp) != e->text_bytes) {
        free(text);
        return NULL;
    }
    text[e->text_bytes] = '\0';
    return text;
}

/* The key bytes of a state: the file text's prefix, or its own. */
static char *kv_read_key(FILE *fp, const ds4_kvstore_entry *e, const ds4_kvstore_state *st) {
    const uint64_t at = st->key_offset ? st->key_offset : e->text_offset;
    if (at > (uint64_t)INT64_MAX || fseeko(fp, (off_t)at, SEEK_SET) != 0) return NULL;
    char *key = kv_xmalloc((size_t)st->key_len + 1);
    if (fread(key, 1, st->key_len, fp) != st->key_len) {
        free(key);
        return NULL;
    }
    key[st->key_len] = '\0';
    return key;
}

/* A checkpoint file this build cannot read (an older layout, a different
 * payload ABI, a write that did not finish) only takes up the budget. */
static bool kv_file_is_ours(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    uint8_t h[4];
    const bool ours = fread(h, 1, sizeof(h), fp) == sizeof(h) &&
                      h[0] == KV_CACHE_MAGIC0 && h[1] == KV_CACHE_MAGIC1 && h[2] == KV_CACHE_MAGIC2;
    fclose(fp);
    return ours;
}

static void kv_cache_refresh(ds4_kvstore *kc) {
    if (!kc->enabled) return;
    ds4_kvstore_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(kc->dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            kv_cache_push(kc, e);
        } else if (kv_file_is_ours(path) && unlink(path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache removed unreadable file %s", kv_log_name(kc), path);
        }
        free(path);
    }
    closedir(d);
}

bool ds4_kvstore_touch_file(const char *path, uint32_t hits, uint64_t used_at) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    bool ok = kv_read_header(fp, &e);
    if (ok) {
        e.hits = hits;
        e.last_used = used_at ? used_at : (uint64_t)time(NULL);
        ok = kv_write_header(fp, &e);
    }
    fclose(fp);
    return ok;
}

/* =========================================================================
 * Eviction
 * ========================================================================= */

/* Eviction is by recency: the file that was loaded (or, never loaded,
 * written) longest ago goes first, whatever it once was hit for.  A
 * conversation that is not coming back leaves its files behind for good,
 * and past hits say nothing about that; the workload that matters is the
 * conversations being switched between now, whose files are the recent
 * ones.  Among files of the same moment the shorter one goes first: it is
 * the cheaper to prefill again. */
double ds4_kvstore_entry_eviction_score(const ds4_kvstore_entry *e) {
    if (!e || e->file_size == 0) return 0.0;
    const uint64_t active_at = e->last_used ? e->last_used : e->created_at;
    return (double)active_at + (double)e->tokens / 4294967296.0;
}

void ds4_kvstore_evict(ds4_kvstore *kc, uint64_t extra_bytes) {
    if (!kc || !kc->enabled || kc->budget_bytes == 0) return;
    if (extra_bytes > kc->budget_bytes) return;
    kv_cache_refresh(kc);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    const uint64_t target = kc->budget_bytes - extra_bytes;
    while (total > target && kc->len > 0) {
        int victim = 0;
        double victim_score = ds4_kvstore_entry_eviction_score(&kc->entry[0]);
        for (int i = 1; i < kc->len; i++) {
            const double score = ds4_kvstore_entry_eviction_score(&kc->entry[i]);
            if (score < victim_score) {
                victim = i;
                victim_score = score;
            }
        }
        ds4_kvstore_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                    kv_log_name(kc),
                    e.tokens,
                    e.hits,
                    (double)e.file_size / (1024.0 * 1024.0),
                    e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        ds4_kvstore_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    kc->log_name = log_name;
    kc->log = log;
    kc->log_ud = log_ud;
    if (!kv_mkdir_p(dir)) {
        kv_logf(kc, DS4_KVSTORE_LOG_DEFAULT,
                "%s: failed to create KV cache directory %s: %s",
                kv_log_name(kc), dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = kv_xstrdup(dir);
    if (budget_mb == 0) budget_mb = DS4_KVSTORE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    ds4_kvstore_evict(kc, 0);
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, eviction=lru)",
            kv_log_name(kc),
            kc->dir,
            (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
            reject_different_quant ? "reject" : "accept",
            kc->opt.min_tokens,
            kc->opt.cold_max_tokens,
            kc->opt.continued_interval_tokens,
            kc->opt.boundary_trim_tokens,
            kc->opt.boundary_align_tokens);
    return true;
}

void ds4_kvstore_close(ds4_kvstore *kc) {
    ds4_kvstore_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

/* =========================================================================
 * Text helpers and store boundaries
 * ========================================================================= */

/* The rendered text of tokens, and where the rendering of each of the
 * `n` prefixes tokens[0..positions[i]) ends. */
static char *kv_render_text(ds4_engine *engine, const ds4_tokens *tokens,
                            const uint32_t *positions, size_t n, size_t *lens,
                            size_t *out_len) {
    kv_buf b = {0};
    for (size_t i = 0; i < n; i++) lens[i] = positions[i] == 0 ? 0 : SIZE_MAX;
    for (int t = 0; t < tokens->len; t++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[t], &len);
        kv_buf_append(&b, piece, len);
        free(piece);
        for (size_t i = 0; i < n; i++) {
            if (positions[i] == (uint32_t)t + 1u) lens[i] = b.len;
        }
    }
    if (out_len) *out_len = b.len;
    return kv_buf_take(&b);
}

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len) {
    return kv_render_text(engine, tokens, NULL, 0, NULL, out_len);
}

bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out) {
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints maximize reuse across independent agent sessions.  The
     * stable rendered chat prefix is everything before the user message that
     * asks this specific task.  Some clients put stable user-role scaffolding
     * first, so use the last user marker before the first assistant marker. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const ds4_kvstore *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens) {
    if (ds4_kvstore_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    ds4_kvstore_note_store(kc, tokens);
    return old;
}

void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc, uint64_t file_bytes,
                                uint64_t *required_bytes_out) {
    /* The serialized size is deterministic for one snapshot.  Reserve 1%
     * headroom so filesystem/accounting surprises cannot produce a file that
     * is immediately removed by the budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes_out) *required_bytes_out = file_bytes + slack;
    if (!kc || kc->budget_bytes == 0) return true;
    return file_bytes + slack <= kc->budget_bytes;
}

static bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                       const char *text,
                                       uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

static bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                             FILE *fp, const char *text,
                             uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

/* =========================================================================
 * Store
 * ========================================================================= */

/* A state about to be written: one of the engine's, or a blob copied from
 * the file being extended (its first state, kept). */
typedef struct {
    uint32_t position;
    uint32_t key_len;
    uint8_t sha[20];
    char *key;              /* the key bytes when they are not the text's prefix */
    uint64_t bytes;
    size_t engine_index;
    uint8_t *blob;          /* when copied */
} kv_state_src;

static uint64_t kv_tail_bytes(const ds4_tokens *tokens, size_t text_len,
                              const kv_state_src *st, uint32_t n, uint64_t trailer_bytes) {
    uint64_t bytes = KV_TAIL_FIXED + (uint64_t)n * KV_TAIL_STATE +
                     (uint64_t)tokens->len * 4u + text_len + trailer_bytes;
    for (uint32_t i = 0; i < n; i++) {
        bytes += st[i].bytes;
        if (st[i].key) bytes += 4u + st[i].key_len;
    }
    return bytes;
}

static bool kv_copy_bytes(FILE *src, uint64_t at, uint64_t bytes, uint8_t *dst) {
    return at <= (uint64_t)INT64_MAX && fseeko(src, (off_t)at, SEEK_SET) == 0 &&
           fread(dst, 1, bytes, src) == bytes;
}

/* The tail at the current position: index, tokens, text, keys, blobs,
 * trailer.  ext_flags gets the trailer's flag when one was written. */
static bool kv_write_tail(FILE *fp, ds4_engine *engine, ds4_session *session,
                          const ds4_tokens *tokens, const char *text, size_t text_len,
                          uint32_t blocks_end, uint32_t block_positions,
                          kv_state_src *st, uint32_t n,
                          const ds4_kvstore_trailer_hooks *hooks, uint8_t *ext_flags,
                          char *err, size_t err_len) {
    (void)engine;
    const off_t start = ftello(fp);
    if (start < 0) return false;
    uint64_t at = (uint64_t)start + KV_TAIL_FIXED + (uint64_t)n * KV_TAIL_STATE +
                  (uint64_t)tokens->len * 4u + text_len;
    uint64_t *key_at = kv_xmalloc((size_t)n * sizeof(uint64_t));
    uint64_t *blob_at = kv_xmalloc((size_t)n * sizeof(uint64_t));
    for (uint32_t i = 0; i < n; i++) {
        key_at[i] = st[i].key ? at : 0;
        if (st[i].key) at += 4u + st[i].key_len;
    }
    for (uint32_t i = 0; i < n; i++) {
        blob_at[i] = at;
        at += st[i].bytes;
    }
    const uint64_t trailer_at = at;

    uint8_t t[KV_TAIL_FIXED];
    memcpy(t, "TAIL", 4);
    ds4_kvstore_le_put32(t + 4, n);
    ds4_kvstore_le_put32(t + 8, blocks_end);
    ds4_kvstore_le_put32(t + 12, block_positions);
    ds4_kvstore_le_put32(t + 16, (uint32_t)text_len);
    ds4_kvstore_le_put32(t + 20, (uint32_t)tokens->len);
    kv_le_put64(t + 24, trailer_at);
    bool ok = fwrite(t, 1, sizeof(t), fp) == sizeof(t);
    for (uint32_t i = 0; ok && i < n; i++) {
        uint8_t r[KV_TAIL_STATE];
        ds4_kvstore_le_put32(r, st[i].position);
        ds4_kvstore_le_put32(r + 4, st[i].key_len);
        memcpy(r + 8, st[i].sha, 20);
        kv_le_put64(r + 28, blob_at[i]);
        kv_le_put64(r + 36, st[i].bytes);
        kv_le_put64(r + 44, key_at[i]);
        ok = fwrite(r, 1, sizeof(r), fp) == sizeof(r);
    }
    for (int i = 0; ok && i < tokens->len; i++) ok = kv_write_u32(fp, (uint32_t)tokens->v[i]);
    if (ok) ok = fwrite(text, 1, text_len, fp) == text_len;
    for (uint32_t i = 0; ok && i < n; i++) {
        if (!st[i].key) continue;
        ok = kv_write_u32(fp, st[i].key_len) && fwrite(st[i].key, 1, st[i].key_len, fp) == st[i].key_len;
    }
    for (uint32_t i = 0; ok && i < n; i++) {
        if (st[i].blob) {
            ok = fwrite(st[i].blob, 1, st[i].bytes, fp) == st[i].bytes;
        } else {
            ok = ds4_session_write_state(session, st[i].engine_index, fp, err, err_len) == 0;
        }
    }
    uint64_t trailer_bytes = 0;
    if (ok) ok = kv_trailer_write(hooks, fp, text, &trailer_bytes);
    if (ok && trailer_bytes > 0 && hooks) *ext_flags |= hooks->ext_flag;
    free(key_at);
    free(blob_at);
    if (!ok && err && err_len && !err[0]) kv_set_err(err, err_len, strerror(errno));
    return ok;
}

/* How the file at path relates to the history about to be stored: it is
 * that history so far (same layout, its history a prefix of the live one,
 * its blocks covering all of it), so the store appends to it; or it holds
 * that history already and more (the session was replayed from an earlier
 * point), so the store has nothing to add; or it is another history. */
typedef enum { KV_FILE_OTHER, KV_FILE_EXTENDS, KV_FILE_COVERS } kv_file_relation;

static kv_file_relation kv_file_relates(ds4_session *session, const char *path,
                                        const ds4_tokens *tokens, int model_id, int quant_bits,
                                        bool reject_quant, ds4_kvstore_entry *e) {
    const uint32_t block_positions = ds4_session_block_positions(session);
    if (!path || block_positions == 0) return KV_FILE_OTHER;
    char sha[41] = {0};
    if (!ds4_kvstore_read_entry_file(path, sha, e)) return KV_FILE_OTHER;
    bool ok = e->model_id == (uint8_t)model_id &&
              (!reject_quant || e->quant_bits == (uint8_t)quant_bits) &&
              e->ctx_size <= (uint32_t)ds4_session_ctx(session) &&
              e->block_positions == block_positions &&
              e->blocks_end == e->tokens &&
              e->tail_offset == DS4_KVSTORE_FIXED_HEADER +
                                ds4_session_block_bytes(session, 0, e->blocks_end);
    const uint32_t common = e->tokens < (uint32_t)tokens->len ? e->tokens : (uint32_t)tokens->len;
    if (ok) {
        FILE *fp = fopen(path, "rb");
        ok = fp != NULL;
        if (ok) ok = e->tokens_offset <= (uint64_t)INT64_MAX &&
                     fseeko(fp, (off_t)e->tokens_offset, SEEK_SET) == 0;
        for (uint32_t i = 0; ok && i < common; i++) {
            uint32_t tok = 0;
            ok = kv_read_u32(fp, &tok) && (int)tok == tokens->v[i];
        }
        if (fp) fclose(fp);
    }
    if (!ok) {
        ds4_kvstore_entry_free(e);
        return KV_FILE_OTHER;
    }
    return e->tokens < (uint32_t)tokens->len ? KV_FILE_EXTENDS : KV_FILE_COVERS;
}

char *ds4_kvstore_store(ds4_kvstore *kc, ds4_engine *engine, ds4_session *session,
                        const ds4_kvstore_store_request *req, char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    if (kc && !kc->enabled) return NULL;
    if (!req || !req->tokens || req->store_len <= 0 || (!kc && !req->path)) return NULL;
    if (kc && req->store_len < kc->opt.min_tokens) return NULL;
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) return NULL;
    const int model_id = ds4_engine_model_id(engine);
    const char *reason = req->reason ? req->reason : "unknown";

    const ds4_tokens *live = ds4_session_tokens(session);
    if (!live || live->len != req->store_len || req->store_len > req->tokens->len ||
        memcmp(live->v, req->tokens->v, (size_t)req->store_len * sizeof(int)) != 0) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                kv_log_name(kc), req->store_len, reason, live ? live->len : -1);
        return NULL;
    }
    ds4_tokens tokens = {0};
    ds4_kvstore_tokens_copy_prefix(&tokens, req->tokens, req->store_len);

    /* the engine's states, the live one first */
    const size_t n_engine = ds4_session_state_count(session);
    if (n_engine == 0 || ds4_session_state_position(session, 0) != (uint32_t)tokens.len) {
        kv_set_err(err, err_len, "session has no state to store");
        ds4_tokens_free(&tokens);
        return NULL;
    }
    uint32_t positions[64];
    size_t lens[64];
    if (n_engine > 63) {
        kv_set_err(err, err_len, "too many saved states");
        ds4_tokens_free(&tokens);
        return NULL;
    }
    for (size_t i = 0; i < n_engine; i++) positions[i] = ds4_session_state_position(session, i);
    size_t text_len = 0;
    char *text = kv_render_text(engine, &tokens, positions, n_engine, lens, &text_len);
    if (text_len > UINT32_MAX) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d because rendered text is too large",
                kv_log_name(kc), tokens.len);
        free(text);
        ds4_tokens_free(&tokens);
        return NULL;
    }
    const bool override = req->key_override && req->key_override[0];
    kv_state_src st[64];
    uint32_t n = 0;
    for (size_t i = 0; i < n_engine; i++) {
        kv_state_src *s = &st[n++];
        memset(s, 0, sizeof(*s));
        s->position = positions[i];
        s->engine_index = i;
        s->bytes = ds4_session_state_bytes(session, i);
        if (i == 0 && override) {
            s->key = kv_xstrdup(req->key_override);
            s->key_len = (uint32_t)strlen(s->key);
            sha1_bytes(s->key, s->key_len, s->sha);
        } else {
            s->key_len = (uint32_t)lens[i];
            sha1_bytes(text, s->key_len, s->sha);
        }
    }

    /* the file to write: the session's own, an explicit path, or a new one */
    const bool reject_quant = kc ? kc->reject_different_quant : true;
    ds4_kvstore_entry old = {0};
    char *path = NULL;
    kv_file_relation relation = KV_FILE_OTHER;
    if (req->path) {
        path = kv_xstrdup(req->path);
        relation = kv_file_relates(session, path, &tokens, model_id, quant_bits, reject_quant, &old);
    } else if (req->extend_path &&
               (relation = kv_file_relates(session, req->extend_path, &tokens, model_id, quant_bits,
                                           reject_quant, &old)) != KV_FILE_OTHER) {
        path = kv_xstrdup(req->extend_path);
    } else {
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
        path = ds4_kvstore_path_for_sha(kc, sha);
        if (access(path, F_OK) == 0) {
            /* the same history so far is one file; another conversation's
             * file may happen to bear this text's name */
            relation = kv_file_relates(session, path, &tokens, model_id, quant_bits, reject_quant, &old);
            if (relation == KV_FILE_OTHER) {
                const uint64_t now = (uint64_t)time(NULL);
                kv_buf b = {0};
                kv_buf_append(&b, text, text_len);
                kv_buf_append(&b, &now, sizeof(now));
                ds4_kvstore_sha1_bytes_hex(b.ptr, b.len, sha);
                free(b.ptr);
                free(path);
                path = ds4_kvstore_path_for_sha(kc, sha);
            }
        }
    }
    if (relation == KV_FILE_COVERS) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache covered tokens=%d reason=%s by tokens=%u file=%s",
                kv_log_name(kc), tokens.len, reason, old.tokens, path);
        for (uint32_t i = 0; i < n; i++) free(st[i].key);
        ds4_kvstore_entry_free(&old);
        free(text);
        ds4_tokens_free(&tokens);
        return path;
    }
    const bool extend = relation == KV_FILE_EXTENDS;
    /* the file's first state stays: the anchor other conversations share */
    if (extend && old.n_states > 0) {
        bool have = false;
        for (uint32_t i = 0; i < n; i++) have |= st[i].position == old.state[0].position;
        if (!have && n < 64) {
            const ds4_kvstore_state *a = &old.state[0];
            FILE *fp = fopen(path, "rb");
            kv_state_src *s = &st[n];
            memset(s, 0, sizeof(*s));
            s->position = a->position;
            s->key_len = a->key_len;
            s->bytes = a->bytes;
            s->blob = kv_xmalloc((size_t)a->bytes);
            bool ok = fp && kv_copy_bytes(fp, a->offset, a->bytes, s->blob);
            if (ok && a->key_offset) {
                s->key = kv_read_key(fp, &old, a);
                ok = s->key != NULL;
            }
            if (fp) fclose(fp);
            if (ok) {
                if (s->key) sha1_bytes(s->key, s->key_len, s->sha);
                else sha1_bytes(text, s->key_len, s->sha);
                n++;
            } else {
                free(s->blob);
                free(s->key);
            }
        }
    }

    uint64_t trailer_est = 0;
    const uint32_t block_positions = ds4_session_block_positions(session);
    const uint64_t blocks_bytes = block_positions ?
        ds4_session_block_bytes(session, 0, (uint32_t)tokens.len) : 0;
    bool ok = kv_trailer_serialized_size(req->hooks, text, &trailer_est);
    const uint64_t new_size = DS4_KVSTORE_FIXED_HEADER + blocks_bytes +
                              kv_tail_bytes(&tokens, text_len, st, n, trailer_est);
    uint64_t required = 0;
    if (ok && !ds4_kvstore_file_size_fits(kc, new_size, &required)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                kv_log_name(kc), tokens.len, reason,
                (double)new_size / (1024.0 * 1024.0),
                (double)required / (1024.0 * 1024.0),
                kc ? (double)kc->budget_bytes / (1024.0 * 1024.0) : 0.0);
        ok = false;
    }
    if (ok && kc) {
        /* the file being extended is in use: it must not be the victim */
        if (extend) ds4_kvstore_touch_file(path, old.hits, 0);
        const uint64_t growth = extend && old.file_size < new_size ? new_size - old.file_size :
                                extend ? 0 : new_size;
        ds4_kvstore_evict(kc, growth);
    }

    const double save_t0 = kv_now_sec();
    char *tmp = NULL;
    FILE *fp = NULL;
    uint32_t from = 0;
    if (ok && extend) {
        fp = fopen(path, "r+b");
        /* the last block may be partial: rewrite it from its start */
        from = old.blocks_end - old.blocks_end % block_positions;
        const uint64_t at = DS4_KVSTORE_FIXED_HEADER + ds4_session_block_bytes(session, 0, from);
        ok = fp && at <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)at, SEEK_SET) == 0;
    } else if (ok) {
        kv_buf b = {0};
        kv_buf_printf(&b, "%s.tmp.%ld", path, (long)getpid());
        tmp = kv_buf_take(&b);
        fp = fopen(tmp, "wb");
        uint8_t zero[DS4_KVSTORE_FIXED_HEADER] = {0};
        ok = fp && fwrite(zero, 1, sizeof(zero), fp) == sizeof(zero);
    }
    if (!ok && fp == NULL && err && err_len && !err[0]) kv_set_err(err, err_len, strerror(errno));
    for (uint32_t b = from; ok && b < (uint32_t)tokens.len; b += block_positions) {
        const uint32_t to = b + block_positions < (uint32_t)tokens.len ? b + block_positions : (uint32_t)tokens.len;
        ok = ds4_session_write_blocks(session, fp, b, to, err, err_len) == 0;
    }
    ds4_kvstore_entry hdr = {0};
    if (ok) {
        const off_t tail = ftello(fp);
        ok = tail >= 0;
        hdr.quant_bits = (uint8_t)quant_bits;
        hdr.model_id = (uint8_t)model_id;
        hdr.reason = ds4_kvstore_reason_code(reason);
        hdr.ext_flags = override ? req->key_ext : 0;
        hdr.tokens = (uint32_t)tokens.len;
        hdr.hits = extend ? old.hits : 0;
        hdr.ctx_size = (uint32_t)ds4_session_ctx(session);
        hdr.created_at = req->created_at ? req->created_at : extend ? old.created_at : (uint64_t)time(NULL);
        hdr.last_used = (uint64_t)time(NULL);
        hdr.tail_offset = (uint64_t)tail;
    }
    if (ok) ok = kv_write_tail(fp, engine, session, &tokens, text, text_len,
                               block_positions ? (uint32_t)tokens.len : 0, block_positions,
                               st, n, req->hooks, &hdr.ext_flags, err, err_len);
    if (ok) {
        const off_t end = ftello(fp);
        ok = end >= 0 && ftruncate(fileno(fp), end) == 0 && kv_write_header(fp, &hdr) && fflush(fp) == 0;
        if (ok) hdr.file_size = (uint64_t)end;
    }
    if (fp && fclose(fp) != 0) ok = false;
    if (ok && tmp && rename(tmp, path) != 0) ok = false;
    if (!ok && err && err_len && !err[0]) kv_set_err(err, err_len, strerror(errno));
    const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
    if (!ok) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache store failed (%s): %s save=%.1f ms",
                kv_log_name(kc), reason, err && err[0] ? err : "unknown error", save_ms);
        if (tmp) unlink(tmp);
    } else {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache %s tokens=%d trimmed=%d reason=%s key=%s states=%u size=%.2f MiB save=%.1f ms file=%s",
                kv_log_name(kc), extend ? "extended" : "stored",
                tokens.len, req->tokens->len - tokens.len, reason,
                override ? (req->key_kind ? req->key_kind : "visible-transcript") : "token-text",
                n, (double)hdr.file_size / (1024.0 * 1024.0), save_ms, path);
        if (kc) ds4_kvstore_note_store(kc, tokens.len);
    }
    for (uint32_t i = 0; i < n; i++) {
        free(st[i].key);
        free(st[i].blob);
    }
    ds4_kvstore_entry_free(&old);
    free(tmp);
    free(text);
    ds4_tokens_free(&tokens);
    if (!ok) {
        free(path);
        return NULL;
    }
    return path;
}

bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       const char *extend_path,
                                       char **path_out,
                                       char *err,
                                       size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens) return false;
    const int target = ds4_kvstore_continued_store_target(kc, tokens->len);
    if (target == 0) return false;
    const ds4_kvstore_store_request req = {
        .tokens = tokens, .store_len = target, .reason = "continued",
        .hooks = hooks, .extend_path = extend_path,
    };
    char *path = ds4_kvstore_store(kc, engine, session, &req, err, err_len);
    if (!path) return false;
    if (path_out) *path_out = path;
    else free(path);
    return true;
}

bool ds4_kvstore_write_text_only(const char *path, uint8_t model_id, uint8_t quant_bits,
                                 uint8_t reason, uint8_t ext_flags, uint32_t tokens,
                                 uint32_t ctx_size, uint64_t created_at, const char *text,
                                 const ds4_kvstore_trailer_hooks *hooks,
                                 char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    const size_t text_len = text ? strlen(text) : 0;
    if (text_len > UINT32_MAX) {
        kv_set_err(err, err_len, "rendered text is too large");
        return false;
    }
    kv_buf b = {0};
    kv_buf_printf(&b, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = kv_buf_take(&b);
    FILE *fp = fopen(tmp, "wb");
    ds4_kvstore_entry hdr = {0};
    hdr.quant_bits = quant_bits;
    hdr.model_id = model_id;
    hdr.reason = reason;
    hdr.ext_flags = ext_flags;
    hdr.tokens = tokens;
    hdr.ctx_size = ctx_size;
    hdr.created_at = created_at ? created_at : (uint64_t)time(NULL);
    hdr.last_used = (uint64_t)time(NULL);
    hdr.tail_offset = DS4_KVSTORE_FIXED_HEADER;
    /* no tokens are kept, the text is the history; the one state, keyed by
     * the text, has no blob: a reader rebuilds from the text */
    ds4_tokens none = {0};
    kv_state_src st = { .position = tokens, .key_len = (uint32_t)text_len, .blob = (uint8_t *)"" };
    sha1_bytes(text ? text : "", text_len, st.sha);
    uint8_t zero[DS4_KVSTORE_FIXED_HEADER] = {0};
    bool ok = fp && fwrite(zero, 1, sizeof(zero), fp) == sizeof(zero) &&
              kv_write_tail(fp, NULL, NULL, &none, text ? text : "", text_len, 0, 0, &st, 1,
                            hooks, &hdr.ext_flags, err, err_len) &&
              kv_write_header(fp, &hdr) && fflush(fp) == 0;
    if (fp && fclose(fp) != 0) ok = false;
    if (ok && rename(tmp, path) != 0) ok = false;
    if (!ok) {
        if (err && err_len && !err[0]) kv_set_err(err, err_len, strerror(errno));
        unlink(tmp);
    }
    free(tmp);
    return ok;
}

/* =========================================================================
 * Lookup and load
 * ========================================================================= */

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size,
                                 uint32_t *state_out) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    uint32_t best_state = 0;
    for (int i = 0; i < kc->len; i++) {
        const ds4_kvstore_entry *e = &kc->entry[i];
        if (e->model_id != (uint8_t)model_id) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        for (uint32_t s = 0; s < e->n_states; s++) {
            const ds4_kvstore_state *st = &e->state[s];
            if (st->key_len > prompt_bytes || (int)st->position < kc->opt.min_tokens) continue;
            if (best >= 0) {
                const ds4_kvstore_state *b = &kc->entry[best].state[best_state];
                if (st->key_len < b->key_len) continue;
                if (st->key_len == b->key_len && st->position <= b->position) continue;
            }
            char sha[41];
            ds4_kvstore_sha1_bytes_hex(prompt_text, st->key_len, sha);
            if (!strcmp(sha, st->sha)) {
                best = i;
                best_state = s;
            }
        }
    }
    if (state_out) *state_out = best_state;
    return best;
}

int ds4_kvstore_load(ds4_engine *engine, ds4_session *session, FILE *fp,
                     const ds4_kvstore_entry *e, uint32_t state_index,
                     const ds4_kvstore_trailer_hooks *hooks,
                     char *err, size_t err_len) {
    (void)engine;
    if (err && err_len) err[0] = '\0';
    if (state_index >= e->n_states) {
        kv_set_err(err, err_len, "no such state in the KV checkpoint");
        return 0;
    }
    const ds4_kvstore_state *st = &e->state[state_index];
    const uint32_t position = st->position;
    if (position == 0 || position > e->tokens || position >= (uint32_t)ds4_session_ctx(session)) {
        kv_set_err(err, err_len, "KV checkpoint state does not fit current context");
        return 0;
    }
    const uint32_t block_positions = ds4_session_block_positions(session);
    if (e->block_positions != block_positions ||
        (block_positions && (position > e->blocks_end ||
                             e->tail_offset != DS4_KVSTORE_FIXED_HEADER +
                                               ds4_session_block_bytes(session, 0, e->blocks_end)))) {
        kv_set_err(err, err_len, "KV checkpoint was written for a different layout");
        return 0;
    }
    int *tokens = kv_xmalloc((size_t)position * sizeof(int));
    bool ok = e->tokens_offset <= (uint64_t)INT64_MAX &&
              fseeko(fp, (off_t)e->tokens_offset, SEEK_SET) == 0;
    for (uint32_t i = 0; ok && i < position; i++) {
        uint32_t tok = 0;
        ok = kv_read_u32(fp, &tok);
        tokens[i] = (int)tok;
    }
    if (!ok) kv_set_err(err, err_len, "truncated KV checkpoint tokens");
    for (uint32_t from = 0; ok && block_positions && from < position; from += block_positions) {
        const uint32_t stored_to = from + block_positions < e->blocks_end ? from + block_positions : e->blocks_end;
        const uint32_t to = stored_to < position ? stored_to : position;
        const uint64_t at = DS4_KVSTORE_FIXED_HEADER + ds4_session_block_bytes(session, 0, from);
        ok = at <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)at, SEEK_SET) == 0 &&
             ds4_session_read_blocks(session, fp, from, to, stored_to, err, err_len) == 0;
    }
    if (ok) {
        ok = st->offset <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)st->offset, SEEK_SET) == 0 &&
             ds4_session_read_state(session, fp, tokens, position, st->bytes, true, err, err_len) == 0;
    }
    /* the earlier states of the same history come along as fallbacks */
    for (uint32_t s = 0; ok && block_positions && s < e->n_states; s++) {
        const ds4_kvstore_state *o = &e->state[s];
        if (s == state_index || o->position >= position || o->position == 0) continue;
        ok = o->offset <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)o->offset, SEEK_SET) == 0 &&
             ds4_session_read_state(session, fp, tokens, o->position, o->bytes, false, err, err_len) == 0;
    }
    if (ok) {
        const ds4_tokens *live = ds4_session_tokens(session);
        ok = live && live->len == (int)position;
        if (!ok) kv_set_err(err, err_len, "KV checkpoint restored the wrong history");
    }
    if (ok && hooks && hooks->load && (e->ext_flags & hooks->ext_flag)) {
        if (e->trailer_offset <= (uint64_t)INT64_MAX && fseeko(fp, (off_t)e->trailer_offset, SEEK_SET) == 0) {
            hooks->load(hooks->ud, fp, hooks->load_wanted);
        }
    }
    free(tokens);
    if (!ok) {
        ds4_session_invalidate(session);
        if (err && err_len && !err[0]) kv_set_err(err, err_len, "failed to load KV checkpoint");
        return 0;
    }
    return (int)position;
}

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    if (result) memset(result, 0, sizeof(*result));
    if (effective_prompt) effective_prompt->len = 0;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) return 0;
    const int model_id = ds4_engine_model_id(engine);
    const size_t prompt_bytes = strlen(prompt_text);
    uint32_t si = 0;
    const int idx = ds4_kvstore_find_text_prefix(kc, prompt_text, model_id, quant_bits,
                                                 ds4_session_ctx(session), &si);
    if (idx < 0) return 0;

    const ds4_kvstore_entry *e = &kc->entry[idx];
    const ds4_kvstore_state *st = &e->state[si];
    char *path = kv_xstrdup(e->path);
    const double load_t0 = kv_now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    /* the hash chose the state; the bytes themselves settle it */
    char *key = kv_read_key(fp, e, st);
    char err[160] = {0};
    int loaded = 0;
    if (!key || !ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes, key, st->key_len)) {
        snprintf(err, sizeof(err), "cached key does not begin the prompt");
    } else {
        loaded = ds4_kvstore_load(engine, session, fp, e, si, hooks, err, sizeof(err));
    }
    fclose(fp);
    if (loaded > 0 && effective_prompt) {
        /* The cache lookup was by bytes, but the graph state is still the
         * exact token history stored in the payload.  Build the prompt from
         * that exact history and tokenize only the text suffix after the
         * key bytes. */
        ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
            engine, ds4_session_tokens(session), prompt_text + st->key_len, effective_prompt);
    }
    if (loaded > 0) {
        const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
        kc->continued_last_store_tokens = loaded;
        ds4_kvstore_touch_file(path, e->hits + 1, 0);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                loaded, st->key_len, e->quant_bits, ds4_kvstore_key_kind(e->ext_flags), load_ms, path);
        if (result) {
            result->tokens = loaded;
            result->key_len = st->key_len;
            result->quant_bits = e->quant_bits;
            result->ext_flags = e->ext_flags;
            result->load_ms = load_ms;
            result->path = kv_xstrdup(path);
        }
    } else {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache load failed%s%s %s: %s load=%.1f ms",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                path, err, (kv_now_sec() - load_t0) * 1000.0);
    }
    free(key);
    free(path);
    return loaded;
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}
