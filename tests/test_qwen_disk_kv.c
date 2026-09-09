/* Qwen disk checkpoint (qwen_session_save_payload / load): a session saved
 * to a file and loaded into a fresh session must continue exactly as the
 * original, including its saved prompt states.
 *
 * Session A: P1 -> generate -> P2 -> generate, then its payload goes to a
 * file.  Session R loads it.  Both then take P3 (an extension: the live
 * state), then B (P2 with a hole: resumes from P1's saved state), and the
 * logits and generated tokens must match bit for bit.
 *
 * usage: test_qwen_disk_kv MAIN.gguf [--mtp MTP.gguf] TEXT [--tokens N] */
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int first;
} progress_state;

static void on_progress(void *ud, const char *event, int current, int total) {
    progress_state *p = ud;
    (void)total;
    if (strcmp(event, "prefill_chunk") == 0 && p->first < 0) p->first = current;
}

static int sync_prompt(ds4_session *s, progress_state *p, const ds4_tokens *prompt, const char *name) {
    char err[256] = {0};
    p->first = -1;
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s: sync failed: %s\n", name, err);
        return 0;
    }
    printf("%s: %d tokens, prefill reported from %d\n", name, prompt->len, p->first);
    return 1;
}

static int argmax(const float *logits) {
    int best = 0;
    for (int i = 1; i < 248320; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static int generate(ds4_session *s, float *logits, int *out, int n, const char *name) {
    char err[256] = {0};
    int have = 0;
    ds4_session_copy_logits(s, logits, 248320);
    while (have < n) {
        int accepted[32];
        const int count = ds4_session_eval_speculative_argmax(s, argmax(logits), n - have, -1,
                                                              accepted, 32, err, sizeof(err));
        if (count <= 0) {
            fprintf(stderr, "%s: generation failed: %s\n", name, err);
            return 0;
        }
        for (int i = 0; i < count && have < n; i++) out[have++] = accepted[i];
        ds4_session_copy_logits(s, logits, 248320);
    }
    return 1;
}

static void append(ds4_tokens *dst, const int *src, int n) {
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src[i]);
}

/* Same prompt on both sessions, then the same generation: everything must agree. */
static int both(ds4_session *a, ds4_session *r, progress_state *pa, progress_state *pr,
                const ds4_tokens *prompt, const char *name, float *la, float *lr) {
    int ga[16], gr[16];
    char na[64], nr[64];
    snprintf(na, sizeof(na), "A %s", name);
    snprintf(nr, sizeof(nr), "R %s", name);
    if (!sync_prompt(a, pa, prompt, na) || !sync_prompt(r, pr, prompt, nr)) return 0;
    if (pa->first != pr->first) {
        fprintf(stderr, "FAIL: %s resumed from different positions (%d and %d)\n", name, pa->first, pr->first);
        return 0;
    }
    if (!generate(a, la, ga, 16, na) || !generate(r, lr, gr, 16, nr)) return 0;
    ds4_session_copy_logits(a, la, 248320);
    ds4_session_copy_logits(r, lr, 248320);
    if (memcmp(ga, gr, sizeof(ga)) || memcmp(la, lr, 248320 * sizeof(float))) {
        fprintf(stderr, "FAIL: %s: the loaded session diverged (tokens %s, logits %s)\n", name,
                memcmp(ga, gr, sizeof(ga)) ? "differ" : "same", memcmp(la, lr, 248320 * sizeof(float)) ? "differ" : "same");
        return 0;
    }
    printf("%s: both sessions generated %d %d %d ... and agree\n", name, ga[0], ga[1], ga[2]);
    return 1;
}

int main(int argc, char **argv) {
    const char *model = NULL, *mtp = NULL, *text_path = NULL;
    int n_tokens = 12000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp") && i + 1 < argc) mtp = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (!model) model = argv[i];
        else text_path = argv[i];
    }
    if (!model || !text_path) {
        fprintf(stderr, "usage: %s MAIN.gguf [--mtp MTP.gguf] TEXT [--tokens N]\n", argv[0]);
        return 2;
    }
    FILE *fp = fopen(text_path, "rb");
    if (!fp) {
        perror(text_path);
        return 2;
    }
    char *text = malloc(4u << 20);
    const size_t text_len = fread(text, 1, (4u << 20) - 1u, fp);
    fclose(fp);
    text[text_len] = 0;

    ds4_engine_options options = {0};
    options.model_path = model;
    options.mtp_path = mtp;
    options.backend = DS4_BACKEND_CUDA;
    options.prefill_chunk = 4096;
    options.mtp_draft_tokens = mtp ? 3 : 0;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;

    ds4_tokens all = {0};
    ds4_tokenize_text(engine, text, &all);
    if (all.len < n_tokens) n_tokens = all.len;
    const int n1 = n_tokens / 3, n2 = n_tokens * 2 / 3;
    const int hole = n1 + (n2 - n1) / 2, hole_len = (n2 - n1) / 8;

    const int ctx = n_tokens + 256;
    ds4_session *a = NULL, *r = NULL;
    if (ds4_session_create(&a, engine, ctx) != 0 || ds4_session_create(&r, engine, ctx) != 0) {
        fprintf(stderr, "session create failed\n");
        return 1;
    }
    progress_state pa = {-1}, pr = {-1};
    ds4_session_set_progress(a, on_progress, &pa);
    ds4_session_set_progress(r, on_progress, &pr);
    float *la = malloc(248320 * sizeof(float));
    float *lr = malloc(248320 * sizeof(float));
    int g1[16], g2[16];

    ds4_tokens p = {0};
    append(&p, all.v, n1);
    if (!sync_prompt(a, &pa, &p, "A P1") || !generate(a, la, g1, 16, "A P1")) return 1;
    append(&p, g1, 16);
    append(&p, all.v + n1, n2 - n1);
    if (!sync_prompt(a, &pa, &p, "A P2") || !generate(a, la, g2, 16, "A P2")) return 1;
    const int p2_len = p.len;

    /* to disk and back */
    char err[256] = {0};
    const char *path = "/tmp/ds4-test-qwen-disk-kv.bin";
    const uint64_t expect = ds4_session_payload_bytes(a);
    FILE *out = fopen(path, "wb");
    if (!out || ds4_session_save_payload(a, out, err, sizeof(err)) != 0) {
        fprintf(stderr, "save failed: %s\n", err);
        return 1;
    }
    const long written = ftell(out);
    fclose(out);
    printf("payload: %.1f MiB written, %.1f MiB announced\n", written / 1048576.0, expect / 1048576.0);
    if ((uint64_t)written != expect) {
        fprintf(stderr, "FAIL: payload size mismatch\n");
        return 1;
    }
    FILE *in = fopen(path, "rb");
    if (!in || ds4_session_load_payload(r, in, (uint64_t)written, err, sizeof(err)) != 0) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    fclose(in);
    unlink(path);
    const ds4_tokens *ra = ds4_session_tokens(a), *rr = ds4_session_tokens(r);
    if (rr->len != ra->len || memcmp(rr->v, ra->v, (size_t)ra->len * sizeof(int))) {
        fprintf(stderr, "FAIL: loaded token history differs\n");
        return 1;
    }
    ds4_session_copy_logits(a, la, 248320);
    ds4_session_copy_logits(r, lr, 248320);
    if (memcmp(la, lr, 248320 * sizeof(float))) {
        fprintf(stderr, "FAIL: loaded logits differ\n");
        return 1;
    }
    printf("loaded: %d tokens, logits identical\n", rr->len);

    /* P3 extends the live state on both; B (P2 with a hole) resumes from P1's saved state on both */
    append(&p, g2, 16);
    append(&p, all.v + n2, n_tokens - n2);
    if (!both(a, r, &pa, &pr, &p, "P3", la, lr)) return 1;
    ds4_tokens b = {0};
    append(&b, all.v, n1);
    append(&b, g1, 16);
    append(&b, all.v + n1, hole - n1);
    append(&b, all.v + hole + hole_len, n2 - hole - hole_len);
    if (!both(a, r, &pa, &pr, &b, "B", la, lr)) return 1;
    if (pa.first <= n1 + 16 || pa.first > n1 + 16 + 4096) {
        fprintf(stderr, "FAIL: B did not resume from P1's saved state (first prefill position %d, P1 ends at %d)\n",
                pa.first, n1 + 16);
        return 1;
    }
    (void)p2_len;
    printf("PASS\n");
    ds4_session_free(a);
    ds4_session_free(r);
    ds4_engine_close(engine);
    return 0;
}
