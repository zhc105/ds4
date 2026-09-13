#ifndef DS4_NGRAM_CACHE_H
#define DS4_NGRAM_CACHE_H

/* Row cache for the Qwen PLE n-gram table: fixed-size rows of a large file,
 * addressed by row id, read with direct I/O on a cache miss.  One service
 * thread owns the cache; callers hand it a pass's row ids and get the row
 * bytes back, and may hint rows they will ask for soon.  The cache knows
 * nothing about the row encoding. */

#include <stdbool.h>
#include <stdint.h>

typedef struct ds4_ngram_cache ds4_ngram_cache;

enum { DS4_NGRAM_CACHE_SMALL_PASS = 256 };

typedef struct {
    uint64_t lookups;         /* rows asked for by gather */
    uint64_t reads;           /* of those, read from disk (the rest were hits) */
    uint64_t prefetch_reads;  /* rows read ahead of a gather */
    uint64_t rereads;         /* reads of a row read before: what a larger cache saves */
    /* Gather calls and the time callers spent blocked in them, which is what
     * the table costs the forward pass.  [0]: passes of at most
     * DS4_NGRAM_CACHE_SMALL_PASS rows (decode, speculative verify, batched
     * decode); [1]: larger ones (prefill chunks). */
    uint64_t gathers[2];
    uint64_t wait_us[2];
    uint64_t slots;           /* capacity in rows */
    uint64_t slots_used;
    uint64_t bytes;           /* data, index and per-slot metadata */
} ds4_ngram_cache_stats;

/* Rows start at data_offset and are row_bytes each.  Returns NULL, after a
 * message on stderr, when the file cannot back the table. */
ds4_ngram_cache *ds4_ngram_cache_open(const char *path, uint64_t data_offset, uint64_t rows,
                                      uint32_t row_bytes, uint64_t cache_bytes);
void ds4_ngram_cache_close(ds4_ngram_cache *c);

/* Copy row rows[i] to out + i * row_bytes, reading what the cache lacks.
 * Blocks until done.  Fails on an I/O error or when the pass holds more
 * distinct rows than the cache has slots. */
bool ds4_ngram_cache_gather(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n, uint8_t *out);

/* Rows that will be gathered soon: read in the background, replacing any
 * earlier hint not yet served.  Returns at once. */
void ds4_ngram_cache_prefetch(ds4_ngram_cache *c, const uint32_t *rows, uint64_t n);

void ds4_ngram_cache_stats_get(ds4_ngram_cache *c, ds4_ngram_cache_stats *out);

#endif
