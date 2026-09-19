/* Disk KV store (ds4_kvstore.c) against a fake engine and session.
 *
 * Pure C: no model, no GPU.  The fake session keeps blocks and states the
 * way the Qwen graph does (blocks for every position, states only at some),
 * and each token renders as one letter, so a history and its text are easy
 * to write down.  The property checked is the one the server relies on: a
 * store at position N leaves a file that a prompt beginning with that
 * history resumes at N.
 *
 * usage: test_kvstore [DIR]   (a scratch directory, default /tmp) */

#include "ds4.h"
#include "ds4_kvstore.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        fprintf(stderr, "  FAIL: %s (line %d)\n", #cond, __LINE__);         \
        g_failed++;                                                         \
    }                                                                       \
} while (0)

/* ---- fake engine and session ------------------------------------------ */

enum { FAKE_BLOCK = 4, FAKE_STATE_BYTES = 16 };

struct ds4_engine { int unused; };
struct ds4_session {
    ds4_tokens tokens;       /* the live history */
    uint32_t saved[8];       /* positions of the saved states */
    size_t n_saved;
    ds4_vision_identity images[4];   /* the pictures of the live history */
    size_t n_images;
};

/* A picture in a fake history is '<', one '#' per placeholder, '>'; its
 * marker spells the first byte of its fingerprint. */
const ds4_vision_identity *ds4_session_vision_identities(const ds4_session *s, size_t *count) {
    *count = s->n_images;
    return s->n_images ? s->images : NULL;
}
void ds4_engine_vision_block(ds4_engine *e, const ds4_vision_identity *image, int *start, int *end) {
    (void)e;
    *start = (int)image->token_start - 1;
    *end = (int)(image->token_start + image->token_count) + 1;
}
void ds4_vision_marker(const uint8_t fingerprint[32], char out[DS4_VISION_MARKER_BYTES]) {
    snprintf(out, DS4_VISION_MARKER_BYTES, "{%02x}", fingerprint[0]);
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 16;
        tv->v = realloc(tv->v, (size_t)tv->cap * sizeof(int));
    }
    tv->v[tv->len++] = token;
}
void ds4_tokens_free(ds4_tokens *tv) { free(tv->v); memset(tv, 0, sizeof(*tv)); }
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e;
    char *s = malloc(2);
    s[0] = (char)('a' + token);
    s[1] = '\0';
    if (len) *len = 1;
    return s;
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;
    out->len = 0;
    for (const char *p = text; *p; p++) ds4_tokens_push(out, *p - 'a');
}

int ds4_engine_model_id(ds4_engine *e) { (void)e; return 1; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }
int ds4_session_ctx(ds4_session *s) { (void)s; return 4096; }
void ds4_session_invalidate(ds4_session *s) { s->tokens.len = 0; s->n_saved = 0; }
const ds4_tokens *ds4_session_tokens(ds4_session *s) { return &s->tokens; }

uint32_t ds4_session_block_positions(ds4_session *s) { (void)s; return FAKE_BLOCK; }
uint64_t ds4_session_block_bytes(ds4_session *s, uint32_t from, uint32_t to) {
    (void)s;
    return (uint64_t)(to - from) * sizeof(int);
}
/* a position's block row is its token, so a restore can be checked */
int ds4_session_write_blocks(ds4_session *s, FILE *fp, uint32_t from, uint32_t to,
                             char *err, size_t errlen) {
    (void)err; (void)errlen;
    return fwrite(s->tokens.v + from, sizeof(int), to - from, fp) == to - from ? 0 : 1;
}
int ds4_session_read_blocks(ds4_session *s, FILE *fp, uint32_t from, uint32_t to,
                            uint32_t stored_to, char *err, size_t errlen) {
    (void)err; (void)errlen;
    if (from == 0) ds4_session_invalidate(s);
    int row;
    for (uint32_t i = from; i < stored_to; i++) {
        if (fread(&row, sizeof(row), 1, fp) != 1) return 1;
        if (i < to) ds4_tokens_push(&s->tokens, row);
    }
    return 0;
}

size_t ds4_session_state_count(ds4_session *s) { return 1 + s->n_saved; }
uint32_t ds4_session_state_position(ds4_session *s, size_t i) {
    return i == 0 ? (uint32_t)s->tokens.len : s->saved[i - 1];
}
uint64_t ds4_session_state_bytes(ds4_session *s, size_t i) { (void)s; (void)i; return FAKE_STATE_BYTES; }
int ds4_session_write_state(ds4_session *s, size_t i, FILE *fp, char *err, size_t errlen) {
    (void)err; (void)errlen;
    uint32_t blob[FAKE_STATE_BYTES / 4] = { ds4_session_state_position(s, i), 0x57a7e };
    return fwrite(blob, sizeof(blob), 1, fp) == 1 ? 0 : 1;
}
int ds4_session_read_state(ds4_session *s, FILE *fp, const int *tokens, uint32_t position,
                           const ds4_vision_identity *images, size_t image_count,
                           uint64_t bytes, bool live, char *err, size_t errlen) {
    (void)err; (void)errlen;
    uint32_t blob[FAKE_STATE_BYTES / 4];
    if (bytes != sizeof(blob) || fread(blob, sizeof(blob), 1, fp) != 1) return 1;
    if (blob[0] != position || blob[1] != 0x57a7e) return 1;
    if (!live) {
        if (s->n_saved < 8) s->saved[s->n_saved++] = position;
        return 0;
    }
    s->tokens.len = 0;
    for (uint32_t i = 0; i < position; i++) ds4_tokens_push(&s->tokens, tokens[i]);
    s->n_images = 0;   /* the state's pictures: the history's that begin inside it */
    for (size_t i = 0; i < image_count && images[i].token_start < position && s->n_images < 4; i++)
        s->images[s->n_images++] = images[i];
    return 0;
}

/* ---- helpers ----------------------------------------------------------- */

static void session_set(ds4_session *s, const char *text) {
    ds4_tokenize_rendered_chat(NULL, text, &s->tokens);
    s->n_saved = 0;
    s->n_images = 0;
}

/* the picture whose placeholders start at token_start, named by one byte */
static void session_add_image(ds4_session *s, uint32_t token_start, uint32_t token_count, uint8_t name) {
    ds4_vision_identity *im = &s->images[s->n_images++];
    memset(im, 0, sizeof(*im));
    im->token_start = token_start;
    im->token_count = token_count;
    im->fingerprint[0] = name;
}

static char *store(ds4_kvstore *kc, ds4_session *s, const char *extend_path, const char *reason) {
    char err[160] = {0};
    const ds4_kvstore_store_request req = {
        .tokens = &s->tokens,
        .store_len = s->tokens.len,
        .reason = reason,
        .extend_path = extend_path,
    };
    char *path = ds4_kvstore_store(kc, NULL, s, &req, err, sizeof(err));
    if (!path) fprintf(stderr, "  store failed: %s\n", err);
    return path;
}

/* the position a prompt resumes at, loaded into a fresh session */
static int resume(ds4_kvstore *kc, const char *prompt) {
    ds4_session fresh = {0};
    ds4_tokens effective = {0};
    ds4_kvstore_load_result lr = {0};
    const int got = ds4_kvstore_try_load_text(kc, NULL, &fresh, prompt, &effective,
                                              &lr, NULL, false);
    if (got > 0) CHECK(fresh.tokens.len == got);
    ds4_kvstore_load_result_free(&lr);
    ds4_tokens_free(&effective);
    ds4_tokens_free(&fresh.tokens);
    return got;
}

static int count_files(const char *dir) {
    DIR *d = opendir(dir);
    int n = 0;
    for (struct dirent *de; d && (de = readdir(d));) n += strstr(de->d_name, ".kv") != NULL;
    if (d) closedir(d);
    return n;
}

static void clear_dir(const char *dir) {
    DIR *d = opendir(dir);
    char path[1024];
    for (struct dirent *de; d && (de = readdir(d));) {
        if (de->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        unlink(path);
    }
    if (d) closedir(d);
}

/* ---- tests ------------------------------------------------------------- */

#define SHARED "systemprompttools"        /* 17 tokens both conversations share */

/* A slot's file after an eviction is the previous conversation's.  The next
 * conversation shares its system prompt, so its early checkpoint's tokens
 * are a prefix of that file, but the file has no state there: the store
 * must fork a file of its own instead of calling the position covered. */
static void test_shared_prefix_checkpoint_forks(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    ds4_session s = {0};
    session_set(&s, SHARED "conversationaturnone");
    char *a_file = store(kc, &s, NULL, "evict");
    CHECK(a_file != NULL);
    CHECK(count_files(dir) == 1);

    session_set(&s, SHARED "con");       /* B's checkpoint, inside the shared prefix */
    char *b_file = store(kc, &s, a_file, "continued");
    CHECK(b_file != NULL);
    CHECK(b_file && a_file && strcmp(b_file, a_file) != 0);
    CHECK(count_files(dir) == 2);
    CHECK(resume(kc, SHARED "conversationbgoeson") == (int)strlen(SHARED "con"));

    /* the previous conversation's own file still resumes in full */
    CHECK(resume(kc, SHARED "conversationaturnonemore") ==
          (int)strlen(SHARED "conversationaturnone"));
    free(b_file);
    free(a_file);
    ds4_tokens_free(&s.tokens);
}

/* Storing a history the file can already resume at writes nothing. */
static void test_same_state_is_covered(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    ds4_session s = {0};
    session_set(&s, SHARED "conversation");
    char *first = store(kc, &s, NULL, "cold");
    char *again = store(kc, &s, first, "evict");
    CHECK(first && again && !strcmp(first, again));
    CHECK(count_files(dir) == 1);
    free(again);
    free(first);
    ds4_tokens_free(&s.tokens);
}

/* A conversation that grew out of its file extends it in place. */
static void test_growth_extends_file(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    ds4_session s = {0};
    session_set(&s, SHARED "turnone");
    char *first = store(kc, &s, NULL, "cold");
    session_set(&s, SHARED "turnoneturntwo");
    char *grown = store(kc, &s, first, "continued");
    CHECK(first && grown && !strcmp(first, grown));
    CHECK(count_files(dir) == 1);
    CHECK(resume(kc, SHARED "turnoneturntwoandmore") == (int)strlen(SHARED "turnoneturntwo"));
    free(grown);
    free(first);
    ds4_tokens_free(&s.tokens);
}

/* A conversation's file keeps its earliest state through every extension,
 * so another conversation sharing only its start (the system prompt and
 * tools, where the first checkpoint lands) still resumes there. */
static void test_anchor_survives_extensions(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    ds4_session s = {0};
    session_set(&s, SHARED);
    char *path = store(kc, &s, NULL, "continued");
    const char *turns[] = { SHARED "turnone", SHARED "turnoneturntwo",
                            SHARED "turnoneturntwoturnthree" };
    for (size_t i = 0; i < sizeof(turns) / sizeof(turns[0]); i++) {
        session_set(&s, turns[i]);
        char *grown = store(kc, &s, path, "continued");
        CHECK(grown && path && !strcmp(grown, path));
        free(grown);
    }
    CHECK(count_files(dir) == 1);
    CHECK(resume(kc, SHARED "anotherconversation") == (int)strlen(SHARED));
    free(path);
    ds4_tokens_free(&s.tokens);
}

/* The server's use of the store, driven at random: one slot whose history
 * grows, rewinds to an earlier point and diverges, or is replaced by another
 * conversation that shares a prefix with one seen before (after an eviction
 * the slot is still bound to the previous conversation's file).  Every store
 * passes the slot's file as the one to extend and binds the slot to the path
 * it returns, as kv_cache_store_live_prefix_text does.
 *
 * What the store promises, checked after every step: the history just stored
 * resumes in full, and for every file a prompt that begins with the history
 * of its earliest state (its anchor) or of its latest store resumes at least
 * that far; whatever is resumed is a prefix of the prompt.  Intermediate
 * states may be dropped; those may not. */
typedef struct {
    char *path;
    char anchor[256];
    char last[256];
} model_file;

static unsigned g_rng = 12345;
static unsigned rnd(unsigned n) {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 16) % n;
}

static void append_random(char *hist, size_t cap) {
    size_t len = strlen(hist);
    for (unsigned k = 1 + rnd(6); k > 0 && len + 1 < cap; k--) hist[len++] = (char)('a' + rnd(5));
    hist[len] = '\0';
}

static int resume_checked(ds4_kvstore *kc, const char *history) {
    char prompt[300];
    snprintf(prompt, sizeof(prompt), "%szz", history);
    ds4_session fresh = {0};
    ds4_tokens effective = {0};
    ds4_kvstore_load_result lr = {0};
    const int got = ds4_kvstore_try_load_text(kc, NULL, &fresh, prompt, &effective,
                                              &lr, NULL, false);
    for (int i = 0; got > 0 && i < got; i++) {
        if (fresh.tokens.v[i] != prompt[i] - 'a') {
            fprintf(stderr, "  FAIL: resumed history is not a prefix of the prompt\n");
            g_failed++;
            break;
        }
    }
    ds4_kvstore_load_result_free(&lr);
    ds4_tokens_free(&effective);
    ds4_tokens_free(&fresh.tokens);
    return got;
}

static void test_random_slot_lifecycle(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    enum { STEPS = 400, MAX_FILES = 512 };
    model_file *files = calloc(MAX_FILES, sizeof(files[0]));
    int n_files = 0;
    ds4_session s = {0};
    char hist[256] = "";
    char *slot_path = NULL;
    uint32_t turns[8];          /* the engine's saved prompt states of this history */
    size_t n_turns = 0;
    const int failed_before = g_failed;

    for (int step = 0; step < STEPS && g_failed == failed_before; step++) {
        const unsigned op = strlen(hist) > 200 ? 1 : rnd(10);
        if (op >= 3 || hist[0] == '\0') {
            /* the conversation goes on; the previous prompt end stays saved */
            if (hist[0] && n_turns < 3) turns[n_turns++] = (uint32_t)strlen(hist);
            else if (hist[0]) {
                memmove(turns, turns + 1, 2 * sizeof(turns[0]));
                turns[2] = (uint32_t)strlen(hist);
            }
            append_random(hist, sizeof(hist));
        } else if (op == 0 && n_files > 0) {
            /* another conversation: a prefix of one seen before, then its own */
            const model_file *f = &files[rnd((unsigned)n_files)];
            const char *src = rnd(2) ? f->anchor : f->last;
            const size_t keep = rnd((unsigned)strlen(src) + 1);
            memmove(hist, src, keep);
            hist[keep] = '\0';
            append_random(hist, sizeof(hist));
            n_turns = 0;
        } else {
            /* rewind to an earlier point and diverge */
            hist[rnd((unsigned)strlen(hist)) + 1] = '\0';
            append_random(hist, sizeof(hist));
            n_turns = 0;
        }
        session_set(&s, hist);
        for (size_t i = 0; i < n_turns; i++)
            if (turns[i] < (uint32_t)s.tokens.len) s.saved[s.n_saved++] = turns[i];

        char *path = store(kc, &s, slot_path, "continued");
        if (!path) { g_failed++; break; }
        free(slot_path);
        slot_path = path;
        model_file *f = NULL;
        for (int i = 0; i < n_files; i++) if (!strcmp(files[i].path, path)) f = &files[i];
        if (!f && n_files < MAX_FILES) {
            /* a new file's anchor is the earliest state it was written with */
            uint32_t first = (uint32_t)s.tokens.len;
            for (size_t i = 0; i < s.n_saved; i++) if (s.saved[i] < first) first = s.saved[i];
            f = &files[n_files++];
            f->path = strdup(path);
            snprintf(f->anchor, sizeof(f->anchor), "%.*s", (int)first, hist);
        }
        if (f && strlen(hist) > strlen(f->last)) snprintf(f->last, sizeof(f->last), "%s", hist);
        if (resume_checked(kc, hist) < (int)strlen(hist)) {
            fprintf(stderr, "  FAIL step %d: the history just stored '%s' is not resumable\n",
                    step, hist);
            g_failed++;
        }

        for (int i = 0; i < n_files; i++) {
            const model_file *m = &files[i];
            if (resume_checked(kc, m->anchor) < (int)strlen(m->anchor)) {
                fprintf(stderr, "  FAIL step %d: anchor '%s' of %s not resumable\n",
                        step, m->anchor, m->path);
                g_failed++;
            }
            if (resume_checked(kc, m->last) < (int)strlen(m->last)) {
                fprintf(stderr, "  FAIL step %d: latest '%s' of %s not resumable\n",
                        step, m->last, m->path);
                g_failed++;
            }
        }
    }
    for (int i = 0; i < n_files; i++) free(files[i].path);
    free(files);
    free(slot_path);
    ds4_tokens_free(&s.tokens);
}

/* A state is its tokens and its pictures.  The placeholders of two pictures
 * of one size are the same tokens, so the key spells a picture by its marker,
 * as a request's text does: the same picture resumes (and the session gets
 * its identity back), another one of the same size does not, and neither
 * does the placeholders' own text. */
static void test_pictures_key_the_state(ds4_kvstore *kc, const char *dir) {
    clear_dir(dir);
    ds4_session s = {0};
    session_set(&s, "system<###>question");
    session_add_image(&s, 7, 3, 0xa1);
    char *path = store(kc, &s, NULL, "cold");
    CHECK(path != NULL);

    ds4_session fresh = {0};
    ds4_kvstore_load_result lr = {0};
    CHECK(ds4_kvstore_try_load_text(kc, NULL, &fresh, "system{a1}questionmore", NULL, &lr, NULL, false) == 19);
    CHECK(fresh.tokens.len == 19 && fresh.n_images == 1 &&
          memcmp(&fresh.images[0], &s.images[0], sizeof(s.images[0])) == 0);
    ds4_kvstore_load_result_free(&lr);
    ds4_tokens_free(&fresh.tokens);
    CHECK(resume(kc, "system{b2}questionmore") == 0);
    CHECK(resume(kc, "system<###>questionmore") == 0);

    /* the same tokens with another picture are another history: the slot's
     * file holds the first picture's rows, so the store forks */
    s.images[0].fingerprint[0] = 0xb2;
    char *other = store(kc, &s, path, "cold");
    CHECK(other != NULL && strcmp(other, path) != 0 && count_files(dir) == 2);
    CHECK(resume(kc, "system{b2}questionmore") == 19);
    CHECK(resume(kc, "system{a1}questionmore") == 19);

    /* growing past the picture extends its own file, and the saved state
     * before the picture resumes a prompt that carries a different one */
    session_set(&s, "system<###>questionanswer");
    session_add_image(&s, 7, 3, 0xb2);
    s.saved[s.n_saved++] = 6;
    char *grown = store(kc, &s, other, "continued");
    CHECK(grown != NULL && strcmp(grown, other) == 0 && count_files(dir) == 2);
    CHECK(resume(kc, "system{b2}questionanswer!") == 25);
    CHECK(resume(kc, "system{c3}question") == 6);

    /* a history that stops inside a picture has no key */
    session_set(&s, "system<##");
    session_add_image(&s, 7, 3, 0xa1);
    const ds4_kvstore_store_request inside = { .tokens = &s.tokens, .store_len = s.tokens.len, .reason = "cold" };
    char err[160] = {0};
    CHECK(ds4_kvstore_store(kc, NULL, &s, &inside, err, sizeof(err)) == NULL && count_files(dir) == 2);

    free(grown);
    free(other);
    free(path);
    ds4_tokens_free(&s.tokens);
}

int main(int argc, char **argv) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/ds4-kvstore-test-%ld", argc > 1 ? argv[1] : "/tmp",
             (long)getpid());
    if (mkdir(dir, 0700) != 0) {
        perror(dir);
        return 1;
    }
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    opt.min_tokens = 1;
    ds4_kvstore kc = {0};
    if (!ds4_kvstore_open(&kc, dir, 1024, true, opt, "test", NULL, NULL)) {
        fprintf(stderr, "cannot open the store in %s\n", dir);
        return 1;
    }
    test_shared_prefix_checkpoint_forks(&kc, dir);
    test_same_state_is_covered(&kc, dir);
    test_growth_extends_file(&kc, dir);
    test_anchor_survives_extensions(&kc, dir);
    test_random_slot_lifecycle(&kc, dir);
    test_pictures_key_the_state(&kc, dir);
    ds4_kvstore_close(&kc);
    clear_dir(dir);
    rmdir(dir);
    if (g_failed) {
        fprintf(stderr, "kvstore tests: %d failure(s)\n", g_failed);
        return 1;
    }
    printf("kvstore tests: ok\n");
    return 0;
}
