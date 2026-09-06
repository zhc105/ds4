/* Qwen turn-boundary state copies (qwen_session_state_save): a prompt that
 * edits the middle of the live history must resume from the last prompt
 * that is still its prefix, with the same numbers as a session that never
 * went past that prompt.
 *
 * Session A: P1 -> generate -> P2 -> generate -> P3 -> generate, then the
 * edited prompt B (P2 + its answer + P3's tail with a hole in it).  Session
 * R: P1 -> generate -> P2, then B directly, which extends its live state.
 * Both prefill B from the end of P2 with identical chunking, so the tokens
 * generated from B and the logits after them must match bit for bit.
 *
 * usage: test_qwen_state_cache MAIN.gguf [--mtp MTP.gguf] TEXT [--tokens N] */
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int first;   /* the first prefill_chunk position reported by the last sync */
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

/* Greedy generation of n tokens through the speculative path. */
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

int main(int argc, char **argv) {
    const char *model = NULL, *mtp = NULL, *text_path = NULL;
    int n_tokens = 20000;
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
    printf("%d text tokens, using %d\n", all.len, n_tokens);
    const int n1 = n_tokens * 2 / 5, n2 = n_tokens * 4 / 5;
    const int hole = n2 + (n_tokens - n2) / 4, hole_len = (n_tokens - n2) / 8;

    const int ctx = n_tokens + 256;
    ds4_session *a = NULL, *r = NULL;
    if (ds4_session_create(&a, engine, ctx) != 0 || ds4_session_create(&r, engine, ctx) != 0) {
        fprintf(stderr, "session create failed\n");
        return 1;
    }
    progress_state pa = {-1}, pr = {-1};
    ds4_session_set_progress(a, on_progress, &pa);
    ds4_session_set_progress(r, on_progress, &pr);
    float *logits_a = malloc(248320 * sizeof(float));
    float *logits_r = malloc(248320 * sizeof(float));
    int g1[16], g2[16], g3[16], ga[16], gr[16];
    int ok = 1;

    /* session A: three growing prompts, each answered */
    ds4_tokens p = {0};
    append(&p, all.v, n1);
    ok = ok && sync_prompt(a, &pa, &p, "A P1") && generate(a, logits_a, g1, 16, "A P1");
    append(&p, g1, 16);
    append(&p, all.v + n1, n2 - n1);
    ok = ok && sync_prompt(a, &pa, &p, "A P2") && generate(a, logits_a, g2, 16, "A P2");
    const int p2_len = p.len;
    ds4_tokens p2 = {0};
    append(&p2, p.v, p2_len);
    append(&p, g2, 16);
    append(&p, all.v + n2, n_tokens - n2);
    ok = ok && sync_prompt(a, &pa, &p, "A P3") && generate(a, logits_a, g3, 16, "A P3");
    if (!ok) return 1;

    /* the edited prompt: P3 with a hole, then P3's answer */
    ds4_tokens b = {0};
    append(&b, p2.v, p2_len);
    append(&b, g2, 16);
    append(&b, all.v + n2, hole - n2);
    append(&b, all.v + hole + hole_len, n_tokens - hole - hole_len);
    append(&b, g3, 16);
    ok = sync_prompt(a, &pa, &b, "A B") && generate(a, logits_a, ga, 16, "A B");
    if (!ok) return 1;
    ds4_session_copy_logits(a, logits_a, 248320);   /* after the generation: the state must stay in step */
    float *logits_b_a = malloc(248320 * sizeof(float));
    memcpy(logits_b_a, logits_a, 248320 * sizeof(float));
    if (pa.first <= p2_len + 16 || pa.first > p2_len + 16 + 4096) {
        fprintf(stderr, "FAIL: B did not resume from P2's copy (first prefill position %d, P2 ends at %d)\n",
                pa.first, p2_len + 16);
        return 1;
    }

    /* session R: the same history without P3, then B as a plain extension */
    ds4_tokens q = {0};
    append(&q, all.v, n1);
    ok = sync_prompt(r, &pr, &q, "R P1") && generate(r, logits_r, gr, 16, "R P1");
    if (!ok) return 1;
    if (memcmp(gr, g1, sizeof(g1))) {
        fprintf(stderr, "FAIL: sessions disagree on P1's answer\n");
        return 1;
    }
    /* R stops at P2 (no answer, or B would extend from a different position
     * with different chunking) and B extends it from there */
    ok = sync_prompt(r, &pr, &p2, "R P2") && sync_prompt(r, &pr, &b, "R B") && generate(r, logits_r, gr, 16, "R B");
    if (!ok) return 1;
    ds4_session_copy_logits(r, logits_r, 248320);

    int fails = 0;
    if (memcmp(ga, gr, sizeof(ga))) {
        fprintf(stderr, "FAIL: tokens generated after B differ\n");
        fails++;
    }
    float max_diff = 0.0f;
    for (int i = 0; i < 248320; i++) {
        const float d = logits_b_a[i] - logits_r[i];
        if (d > max_diff) max_diff = d;
        if (-d > max_diff) max_diff = -d;
    }
    printf("generated after B: A %d %d %d ... R %d %d %d ...; logits max |diff| %g\n",
           ga[0], ga[1], ga[2], gr[0], gr[1], gr[2], max_diff);
    if (max_diff != 0.0f) {
        fprintf(stderr, "FAIL: logits after B differ\n");
        fails++;
    }
    /* an unrelated prompt has no usable copy and starts over */
    ds4_tokens u = {0};
    append(&u, all.v + 1, n1);
    ok = sync_prompt(a, &pa, &u, "A unrelated");
    if (!ok) return 1;
    if (pa.first > 4096) {
        fprintf(stderr, "FAIL: unrelated prompt resumed from a copy (first prefill position %d)\n", pa.first);
        fails++;
    }
    /* P3 again is its own copy: no prefill, the copy's logits */
    ok = sync_prompt(a, &pa, &p, "A P3 again");
    if (!ok) return 1;
    ds4_session_copy_logits(a, logits_a, 248320);
    if (pa.first != -1) {
        fprintf(stderr, "FAIL: P3 again prefilled from %d\n", pa.first);
        fails++;
    }
    if (argmax(logits_a) != g3[0]) {
        fprintf(stderr, "FAIL: P3 again does not restore P3's logits (argmax %d, answer began %d)\n", argmax(logits_a), g3[0]);
        fails++;
    }
    printf("%s\n", fails ? "FAILED" : "PASS");
    ds4_session_free(a);
    ds4_session_free(r);
    ds4_engine_close(engine);
    return fails ? 1 : 0;
}
