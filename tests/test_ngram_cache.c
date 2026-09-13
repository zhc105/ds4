/* Unit tests for the PLE n-gram row cache against a synthetic table.
 *
 * Pure C: no model, no GPU.  The table file goes in the directory named by
 * argv[1] (default "."), so the same test covers direct I/O on a real
 * filesystem and the buffered fallback on tmpfs. */

#include "ds4_ngram_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

enum { RB = 160, HEADER = 4096 };
static const uint64_t ROWS = 100003;   /* odd, so the last row ends off a sector boundary */
static char g_path[4096];
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;

static uint64_t rnd(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

static uint8_t cell(uint64_t row, uint32_t j) {
    uint64_t x = row * 0x9e3779b97f4a7c15ull + j * 0xbf58476d1ce4e5b9ull + 1u;
    x ^= x >> 29;
    x *= 0x94d049bb133111ebull;
    return (uint8_t)(x >> 56);
}

static int make_table(const char *dir) {
    snprintf(g_path, sizeof g_path, "%s/ngram_cache_test.XXXXXX", dir);
    int fd = mkstemp(g_path);
    if (fd < 0) return 0;
    FILE *f = fdopen(fd, "wb");
    static uint8_t buf[RB * 1024];
    memset(buf, 0, HEADER);
    fwrite(buf, 1, HEADER, f);
    for (uint64_t r = 0; r < ROWS; r += 1024) {
        uint64_t n = ROWS - r < 1024 ? ROWS - r : 1024;
        for (uint64_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < RB; j++) buf[i * RB + j] = cell(r + i, j);
        }
        fwrite(buf, RB, n, f);
    }
    return fclose(f) == 0;
}

static int rows_match(const uint32_t *rows, uint64_t n, const uint8_t *out) {
    for (uint64_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < RB; j++) {
            if (out[i * RB + j] != cell(rows[i], j)) return 0;
        }
    }
    return 1;
}

/* A small cache under a skewed stream with repeats inside a pass: exercises
 * CLOCK eviction, re-admission and the table's last row. */
static void test_eviction_stream(void) {
    ds4_ngram_cache *c = ds4_ngram_cache_open(g_path, HEADER, ROWS, RB, 2000 * RB);
    CHECK(c != NULL, "open");
    if (!c) return;
    uint32_t rows[700];
    uint8_t out[700 * RB];
    uint64_t asked = 0;
    int all_ok = 1;
    for (int pass = 0; pass < 400; pass++) {
        uint64_t n = 1 + rnd() % 700;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t k = rnd() % 10;
            rows[i] = k < 5 ? (uint32_t)(rnd() % 3000) :
                      k < 8 ? (uint32_t)(rnd() % ROWS) :
                      k < 9 ? rows[rnd() % (i + 1)] : (uint32_t)(ROWS - 1);
        }
        if (!ds4_ngram_cache_gather(c, rows, n, out) || !rows_match(rows, n, out)) all_ok = 0;
        asked += n;
    }
    CHECK(all_ok, "every gathered row matches the file");
    ds4_ngram_cache_stats st;
    ds4_ngram_cache_stats_get(c, &st);
    CHECK(st.lookups == asked, "lookups count every row asked for");
    CHECK(st.slots_used == 2000, "the cache filled");
    CHECK(st.rereads > 0, "evicted rows were read again");
    ds4_ngram_cache_close(c);
}

static void test_repeat_is_hit(void) {
    ds4_ngram_cache *c = ds4_ngram_cache_open(g_path, HEADER, ROWS, RB, 4096 * RB);
    CHECK(c != NULL, "open");
    if (!c) return;
    uint32_t rows[500];
    uint8_t out[500 * RB];
    for (int i = 0; i < 500; i++) rows[i] = (uint32_t)(rnd() % ROWS);
    ds4_ngram_cache_stats a, b;
    CHECK(ds4_ngram_cache_gather(c, rows, 500, out) && rows_match(rows, 500, out), "first gather");
    ds4_ngram_cache_stats_get(c, &a);
    CHECK(ds4_ngram_cache_gather(c, rows, 500, out) && rows_match(rows, 500, out), "second gather");
    ds4_ngram_cache_stats_get(c, &b);
    CHECK(b.reads == a.reads, "a repeated pass reads nothing");
    ds4_ngram_cache_close(c);
}

/* A pass with more distinct rows than slots fails without poisoning the cache. */
static void test_pass_larger_than_cache(void) {
    ds4_ngram_cache *c = ds4_ngram_cache_open(g_path, HEADER, ROWS, RB, 100 * RB);
    CHECK(c != NULL, "open");
    if (!c) return;
    uint32_t rows[300];
    uint8_t out[300 * RB];
    for (uint32_t i = 0; i < 300; i++) rows[i] = i * 7u;
    CHECK(!ds4_ngram_cache_gather(c, rows, 300, out), "oversized pass fails");
    for (uint32_t i = 0; i < 60; i++) rows[i] = i * 7u;
    CHECK(ds4_ngram_cache_gather(c, rows, 60, out) && rows_match(rows, 60, out), "next pass is correct");
    rows[0] = (uint32_t)ROWS;
    CHECK(!ds4_ngram_cache_gather(c, rows, 1, out), "row id past the table fails");
    ds4_ngram_cache_close(c);
}

/* Read-ahead lands in the cache: once it has drained, gathering the hinted
 * rows reads nothing.  Gathers issued while it runs stay correct. */
static void test_prefetch(void) {
    ds4_ngram_cache *c = ds4_ngram_cache_open(g_path, HEADER, ROWS, RB, 8192 * RB);
    CHECK(c != NULL, "open");
    if (!c) return;
    uint32_t hint[4000], rows[64];
    uint8_t out[64 * RB];
    for (int i = 0; i < 4000; i++) hint[i] = (uint32_t)(50000 + i * 11);
    ds4_ngram_cache_prefetch(c, hint, 4000);
    int all_ok = 1;
    for (int pass = 0; pass < 50; pass++) {
        for (int i = 0; i < 64; i++) rows[i] = (uint32_t)(rnd() % ROWS);
        if (!ds4_ngram_cache_gather(c, rows, 64, out) || !rows_match(rows, 64, out)) all_ok = 0;
    }
    CHECK(all_ok, "gathers during read-ahead are correct");
    ds4_ngram_cache_stats st;
    for (int wait = 0; wait < 500; wait++) {
        ds4_ngram_cache_stats_get(c, &st);
        if (st.prefetch_reads + st.reads >= 4000) break;
        nanosleep(&(struct timespec){ .tv_nsec = 10000000 }, NULL);
    }
    /* hinted rows a racing gather read first count as its reads */
    uint8_t big[4000 * RB];
    ds4_ngram_cache_stats before;
    ds4_ngram_cache_stats_get(c, &before);
    CHECK(ds4_ngram_cache_gather(c, hint, 4000, big) && rows_match(hint, 4000, big), "hinted rows");
    ds4_ngram_cache_stats_get(c, &st);
    CHECK(st.reads == before.reads, "hinted rows were already cached");
    ds4_ngram_cache_close(c);
}

int main(int argc, char **argv) {
    if (!make_table(argc > 1 ? argv[1] : ".")) {
        fprintf(stderr, "cannot write the test table\n");
        return 1;
    }
    RUN(test_eviction_stream);
    RUN(test_repeat_is_hit);
    RUN(test_pass_larger_than_cache);
    RUN(test_prefetch);
    unlink(g_path);
    fprintf(stderr, "%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed ? 1 : 0;
}
