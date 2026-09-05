/* Qwen3-VL image preprocessing: the smart-resize grid decisions and the
 * patch layout, checked without a GPU. */
#include "ds4_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t MIN_PIXELS = 65536u;
static const uint32_t MAX_PIXELS = 16777216u;

static ds4_image make_image(uint32_t width, uint32_t height) {
    ds4_image image = {.width = width, .height = height};
    image.rgb = calloc((size_t)width * height * 3u, 1u);
    if (!image.rgb) exit(1);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *pixel = image.rgb + ((size_t)y * width + x) * 3u;
            pixel[0] = (uint8_t)(x + 3u * y);
            pixel[1] = (uint8_t)(x + 3u * y + 7u);
            pixel[2] = (uint8_t)(x + 3u * y + 14u);
        }
    }
    return image;
}

/* The grid transformers' smart_resize picks for this size. */
static int check_resize(uint32_t width, uint32_t height,
                        uint32_t want_width, uint32_t want_height) {
    ds4_image image = make_image(width, height);
    ds4_image_patches patches = {0};
    char error[160] = {0};
    int ok = ds4_image_preprocess_qwen(&patches, &image, MIN_PIXELS, MAX_PIXELS,
                                       error, sizeof(error));
    if (!ok) fprintf(stderr, "%ux%u: preprocess failed: %s\n", width, height, error);
    else if (patches.content_width != want_width || patches.content_height != want_height ||
             patches.grid_width != want_width / 16u || patches.grid_height != want_height / 16u ||
             patches.image_token_count != patches.grid_width * patches.grid_height / 4u) {
        fprintf(stderr, "%ux%u -> %ux%u, expected %ux%u\n", width, height,
                patches.content_width, patches.content_height, want_width, want_height);
        ok = 0;
    }
    ds4_image_patches_free(&patches);
    free(image.rgb);
    return ok;
}

/* An unresized image: every patch value must be the normalised pixel at the
 * coordinate the 2x2 block order, channel-major kernel and repeated frame
 * imply. */
static int check_layout(void) {
    ds4_image image = make_image(320u, 256u);
    ds4_image_patches patches = {0};
    char error[160] = {0};
    int ok = ds4_image_preprocess_qwen(&patches, &image, MIN_PIXELS, MAX_PIXELS,
                                       error, sizeof(error)) &&
             patches.content_width == 320u && patches.content_height == 256u &&
             patches.patch_count == 320u;
    if (!ok) fprintf(stderr, "layout preprocess failed: %s\n", error);
    const uint32_t grid_w = 20u;
    for (uint32_t p = 0; ok && p < patches.patch_count; p++) {
        const uint32_t block = p / 4u, within = p % 4u;
        const uint32_t py = (block / (grid_w / 2u)) * 2u + within / 2u;
        const uint32_t px = (block % (grid_w / 2u)) * 2u + within % 2u;
        const float *patch = patches.patches + (size_t)p * 1536u;
        for (uint32_t c = 0; ok && c < 3u; c++) {
            for (uint32_t t = 0; ok && t < 2u; t++) {
                for (uint32_t y = 0; ok && y < 16u; y++) {
                    for (uint32_t x = 0; ok && x < 16u; x++) {
                        const uint8_t *pixel = image.rgb +
                            ((size_t)(py * 16u + y) * 320u + px * 16u + x) * 3u;
                        const float want = pixel[c] / 255.0f * 2.0f - 1.0f;
                        const float got = patch[((c * 2u + t) * 16u + y) * 16u + x];
                        if (fabsf(got - want) > 1e-6f) {
                            fprintf(stderr, "patch %u c%u t%u (%u,%u): %g vs %g\n",
                                    p, c, t, x, y, got, want);
                            ok = 0;
                        }
                    }
                }
            }
        }
    }
    ds4_image_patches_free(&patches);
    free(image.rgb);
    return ok;
}

int main(void) {
    int ok = check_resize(640u, 400u, 640u, 384u) &&    /* 12.5 rows round to 12 */
             check_resize(640u, 416u, 640u, 416u) &&
             check_resize(512u, 507u, 512u, 512u) &&
             check_resize(1400u, 900u, 1408u, 896u) &&
             check_resize(20u, 20u, 256u, 256u) &&      /* below the minimum */
             check_resize(10000u, 10000u, 4096u, 4096u) &&   /* above the maximum */
             check_layout();
    if (!ok) return 1;
    puts("qwen vision image: ok");
    return 0;
}
