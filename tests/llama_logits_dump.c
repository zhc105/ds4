/* Teacher-forced logits dump via libllama: reads token ids (one per line or
 * space separated) from a file, runs them through the model on the CPU, and
 * writes one f32 logits row per position to the output file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.txt OUT.bin\n", argv[0]);
        return 2;
    }
    FILE *tf = fopen(argv[2], "r");
    if (!tf) { perror("tokens"); return 1; }
    int cap = 4096, n = 0;
    llama_token *toks = malloc(cap * sizeof(*toks));
    long v;
    while (fscanf(tf, " %ld", &v) == 1) {
        if (n == cap) { cap *= 2; toks = realloc(toks, cap * sizeof(*toks)); }
        toks[n++] = (llama_token)v;
    }
    fclose(tf);
    if (n == 0) { fprintf(stderr, "no tokens\n"); return 1; }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 8;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_threads = 16;
    cp.n_threads_batch = 16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE *out = fopen(argv[3], "wb");
    if (!out) { perror("out"); return 1; }
    for (int i = 0; i < n; i++) {
        const float *logits = llama_get_logits_ith(ctx, i);
        fwrite(logits, sizeof(float), n_vocab, out);
    }
    fclose(out);
    fprintf(stderr, "wrote %d positions x %d logits\n", n, n_vocab);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
