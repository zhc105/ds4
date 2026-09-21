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
        /* as the server stores: a cold or continued state is a checkpoint,
         * an evicted slot's or a shutdown's only the live state */
        .checkpoint = !strcmp(reason, "cold") || !strcmp(reason, "continued"),
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

/* the size of FILE.kv's checkpoint companion, -1 when it has none */
static long companion_bytes(const char *path) {
    char ckpt[1100];
    snprintf(ckpt, sizeof(ckpt), "%.*s.ckpt", (int)strlen(path) - 3, path);
    struct stat st;
    return stat(ckpt, &st) == 0 ? (long)st.st_size : -1;
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
 * grows, rewinds to an earlier point and diverges (the same conversation),
 * or is replaced by another conversation that shares a prefix with one seen
 * before (the slot lets go of the old one's file).  Every store passes the
 * slot's file as the session's own and binds the slot to the path it
 * returns, as kv_cache_store_live_prefix_text does; one in three is a
 * checkpoint.
 *
 * What the store promises, checked after every step: the history just stored
 * resumes in full; for every file a prompt that begins with the history of
 * one of its checkpoints, or with what the file held at its last store,
 * resumes at least that far; whatever is resumed is a prefix of the prompt.
 * A rewind rewrites the conversation's own file from the fork: the branch
 * given up is gone, its checkpoints with it, those before the fork stand,
 * and the companion holds exactly the checkpoints that do.  Only a new
 * conversation adds a file, or a history that leaves its file before the
 * file's first checkpoint, which to the store is one. */
enum { MODEL_CKPTS = 128 };
typedef struct {
    char *path;
    char ckpt[MODEL_CKPTS][256];   /* the histories of its checkpoints, in order */
    int n_ckpt;
    char last[256];                /* what it held at its last store */
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
    uint32_t turn = 0;          /* a turn end the session holds in memory: never the file's business */
    int allowed_files = 1;      /* one per conversation */
    const int failed_before = g_failed;

    for (int step = 0; step < STEPS && g_failed == failed_before; step++) {
        const unsigned op = strlen(hist) > 200 ? 1 : rnd(10);
        if (op >= 3 || hist[0] == '\0') {
            /* the conversation goes on */
            turn = (uint32_t)strlen(hist);
            append_random(hist, sizeof(hist));
        } else if (op == 0 && n_files > 0) {
            /* another conversation: a prefix of one seen before, then its own */
            const model_file *f = &files[rnd((unsigned)n_files)];
            const char *src = rnd(2) ? f->ckpt[0] : f->last;
            const size_t keep = rnd((unsigned)strlen(src) + 1);
            memmove(hist, src, keep);
            hist[keep] = '\0';
            append_random(hist, sizeof(hist));
            turn = 0;
            /* no state of the slot begins it: the slot's file stays the old
             * conversation's and this one gets its own */
            free(slot_path);
            slot_path = NULL;
            allowed_files++;
        } else {
            /* rewind to an earlier point and diverge: the same conversation,
             * resumed in memory, whose file gives up the branch left behind
             * wherever the fork lies past its first checkpoint (before it
             * the two share no state, and the store begins a new file) */
            const size_t keep = rnd((unsigned)strlen(hist)) + 1;
            const model_file *own = NULL;
            for (int i = 0; slot_path && i < n_files; i++) if (!strcmp(files[i].path, slot_path)) own = &files[i];
            if (!own || keep < strlen(own->ckpt[0])) allowed_files++;
            hist[keep] = '\0';
            append_random(hist, sizeof(hist));
            turn = 0;
        }
        session_set(&s, hist);
        if (turn != 0 && turn < (uint32_t)s.tokens.len) s.saved[s.n_saved++] = turn;

        /* a checkpoint now and then, the live state alone otherwise */
        const bool checkpoint = rnd(3) == 0;
        char *path = store(kc, &s, slot_path, checkpoint ? "continued" : "evict");
        if (!path) { g_failed++; break; }
        free(slot_path);
        slot_path = path;
        model_file *f = NULL;
        for (int i = 0; i < n_files; i++) if (!strcmp(files[i].path, path)) f = &files[i];
        if (!f && n_files < MAX_FILES) {
            f = &files[n_files++];
            f->path = strdup(path);
        }
        if (!f) break;
        /* The file already resumes at this history (a checkpoint of it, or
         * where it stood): the store wrote nothing.  Otherwise the file now
         * holds this history: the checkpoints past the fork are gone with
         * the branch, and this state is one when the store said so or the
         * file has none (a file begins with a checkpoint). */
        bool covered = !strcmp(f->last, hist);
        for (int i = 0; i < f->n_ckpt; i++) covered |= !strcmp(f->ckpt[i], hist);
        if (!covered) {
            while (f->n_ckpt > 0 && strncmp(f->ckpt[f->n_ckpt - 1], hist, strlen(f->ckpt[f->n_ckpt - 1])) != 0) f->n_ckpt--;
            if ((checkpoint || f->n_ckpt == 0) && f->n_ckpt < MODEL_CKPTS) {
                snprintf(f->ckpt[f->n_ckpt++], sizeof(f->ckpt[0]), "%s", hist);
            }
            snprintf(f->last, sizeof(f->last), "%s", hist);
        }
        if (resume_checked(kc, hist) < (int)strlen(hist)) {
            fprintf(stderr, "  FAIL step %d: the history just stored '%s' is not resumable\n",
                    step, hist);
            g_failed++;
        }
        /* the companion holds the file's checkpoints and nothing else */
        const long companion = companion_bytes(path);
        if (companion != (long)f->n_ckpt * (16 + FAKE_STATE_BYTES)) {
            fprintf(stderr, "  FAIL step %d: a companion of %ld bytes for %d checkpoints\n",
                    step, companion, f->n_ckpt);
            g_failed++;
        }
        if (count_files(dir) > allowed_files) {
            fprintf(stderr, "  FAIL step %d: %d files for %d conversations\n", step, count_files(dir), allowed_files);
            g_failed++;
        }

        for (int i = 0; i < n_files; i++) {
            const model_file *m = &files[i];
            for (int k = 0; k < m->n_ckpt; k++) {
                if (resume_checked(kc, m->ckpt[k]) >= (int)strlen(m->ckpt[k])) continue;
                fprintf(stderr, "  FAIL step %d: checkpoint '%s' of %s not resumable\n",
                        step, m->ckpt[k], m->path);
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

    /* A conversation checkpointed before its picture: the checkpoint is in
     * the companion, the live state in the tail, and the checkpoint resumes
     * a prompt that carries a different picture. */
    clear_dir(dir);
    session_set(&s, "system");
    char *file = store(kc, &s, NULL, "cold");
    session_set(&s, "system<###>questionanswer");
    session_add_image(&s, 7, 3, 0xb2);
    char *live = store(kc, &s, file, "evict");
    CHECK(file && live && !strcmp(live, file) && count_files(dir) == 1);
    CHECK(resume(kc, "system{b2}questionanswer!") == 25);
    CHECK(resume(kc, "system{c3}question") == 6);
    CHECK(companion_bytes(file) == 16 + FAKE_STATE_BYTES);

    /* the agent strips the picture, which its slot resumed in memory: the
     * same conversation, whose own file is rewritten from where the picture
     * began; no second file appears and the branch given up is gone */
    session_set(&s, "systemstrippedquestionanswer");
    char *stripped = store(kc, &s, file, "continued");
    CHECK(stripped != NULL && strcmp(stripped, file) == 0 && count_files(dir) == 1);
    CHECK(resume(kc, "systemstrippedquestionanswer!") == 28);
    CHECK(resume(kc, "system{b2}questionanswer!") == 6);
    CHECK(companion_bytes(file) == 2 * (16 + FAKE_STATE_BYTES));

    /* an edit behind a checkpoint is the same conversation still: the file
     * is rewritten from the fork, and the checkpoint past it goes with the
     * blocks it stood on while the one before it stands */
    session_set(&s, "systemotherquestion");
    char *edited = store(kc, &s, file, "evict");
    CHECK(edited != NULL && strcmp(edited, file) == 0 && count_files(dir) == 1);
    CHECK(resume(kc, "systemotherquestion!") == 19);
    CHECK(resume(kc, "systemstrippedquestionanswer!") == 6);
    CHECK(companion_bytes(file) == 16 + FAKE_STATE_BYTES);

    /* a history that leaves the file before its first checkpoint shares no
     * state with it: that is another conversation's, whatever the caller
     * says, and the file stays as it is */
    session_set(&s, "sysadmin");
    char *early = store(kc, &s, file, "evict");
    CHECK(early != NULL && strcmp(early, file) != 0 && count_files(dir) == 2);
    CHECK(resume(kc, "sysadmin!") == 8 && resume(kc, "systemotherquestion!") == 19);
    CHECK(companion_bytes(file) == 16 + FAKE_STATE_BYTES);

    /* a history that stops inside a picture has no key */
    session_set(&s, "system<##");
    session_add_image(&s, 7, 3, 0xa1);
    const ds4_kvstore_store_request inside = { .tokens = &s.tokens, .store_len = s.tokens.len, .reason = "cold" };
    char err[160] = {0};
    CHECK(ds4_kvstore_store(kc, NULL, &s, &inside, err, sizeof(err)) == NULL && count_files(dir) == 2);

    /* removing a file takes its companion along */
    CHECK(ds4_kvstore_remove(file) && companion_bytes(file) < 0 && count_files(dir) == 1);

    free(early);
    free(edited);
    free(stripped);
    free(live);
    free(file);
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
