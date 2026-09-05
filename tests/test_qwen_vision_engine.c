/* Encode one image through the Qwen3.8-Flash-Next vision tower and dump the
 * language-model embeddings (and optionally the preprocessed patches) for
 * comparison with the transformers reference (tests/qwen_vision_ref.py). */
#include "ds4.h"
#include "ds4_image.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_f32(const char *path, const float *values, size_t count) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }
    const int ok = fwrite(values, sizeof(float), count, fp) == count && fclose(fp) == 0;
    if (!ok) fprintf(stderr, "cannot write %s\n", path);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s MAIN.gguf VISION.gguf IMAGE OUTPUT.f32 [PATCHES.f32]\n", argv[0]);
        return 2;
    }
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.vision_path = argv[2];
    options.backend = DS4_BACKEND_CUDA;
    options.inspect_only = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    char error[256] = {0};
    ds4_vision_embedding embedding = {0};
    if (!ds4_engine_vision_encode_file(engine, argv[3], &embedding, error, sizeof(error))) {
        fprintf(stderr, "vision encode failed: %s\n", error);
        ds4_engine_close(engine);
        return 1;
    }
    int ok = write_f32(argv[4], embedding.data,
                       (size_t)embedding.token_count * embedding.dim);
    printf("%ux%u -> %ux%u, %u image tokens of %u\n",
           embedding.width, embedding.height,
           embedding.content_width, embedding.content_height,
           embedding.token_count, embedding.dim);
    if (ok && argc == 6) {
        ds4_image image = {0};
        ds4_image_patches patches = {0};
        ok = ds4_image_decode_file(&image, argv[3], error, sizeof(error)) &&
             ds4_image_preprocess_qwen(&patches, &image, 65536u, 16777216u,
                                       error, sizeof(error));
        if (!ok) fprintf(stderr, "patch dump failed: %s\n", error);
        else ok = write_f32(argv[5], patches.patches, (size_t)patches.patch_count * 1536u);
        ds4_image_patches_free(&patches);
        ds4_image_free(&image);
    }
    ds4_vision_embedding_free(&embedding);
    ds4_engine_close(engine);
    return ok ? 0 : 1;
}
