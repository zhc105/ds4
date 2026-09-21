#ifndef DS4_CHAINSTORE_H
#define DS4_CHAINSTORE_H

/* The server's disk KV store: conversations as chains of immutable segments.
 *
 * A conversation's history on disk is a chain of files.  Every file but the
 * last is a sealed segment: the blocks of a run of positions and the state
 * at its end, written once and never again.  The server seals at every
 * multiple of SEGMENT_TOKENS its prefill stops at, so a segment is that
 * long, or a multiple of it where a boundary fell inside a picture or was
 * crossed while generating (no state stood there to seal); the store itself
 * takes whatever run it is given.  A segment names its parent, and is named
 * by what it stands for,
 *
 *     id = sha1(root, the history's rendered text from its beginning to the
 *               segment's end)
 *
 * (root: the model, its routed quantization, the block size, this format;
 * the text spells each picture by its marker, so it tells pictures apart).
 * Two conversations that share a history share its segments: the second one
 * finds the file already there.  Nothing a segment says can become wrong,
 * because only text a client sent is ever sealed (ds4_server.c,
 * server_prompt_sync): what a turn generates is history once it comes back
 * in a prompt, and is sealed by that prompt's prefill.
 *
 * The last file is the tail: the blocks past the last sealed segment and up
 * to two states, the live one where the slot stood when it was given up
 * (generation included, for the client that sends it back as it was) and the
 * one saved where its client's text ended (for the client that does not).
 * A tail is rewritten by the one slot that holds it and replaced by the
 * next; a crash can damage nothing else.
 *
 * A request resumes the longest state whose text begins it.  Every state is
 * (text length, sha1 of that much text), so one pass over the request's
 * text finds it with no file read.  A request that leaves a history inside
 * a segment resumes from the segment before and hangs a new tail there: the
 * branch given up becomes a leaf nobody extends.
 *
 * Files go when the budget asks, leaves first and least recently used
 * first: a segment is never removed from under another, and the chains of
 * live slots are pinned.  That is the only place anything is deleted. */

#include "ds4.h"
#include "ds4_kvstore.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_CHAINSTORE_SEGMENT_TOKENS 16384
#define DS4_CHAINSTORE_ID_BYTES 20

typedef struct {
    uint8_t id[DS4_CHAINSTORE_ID_BYTES];        /* a sealed segment's name; a tail's live state */
    uint8_t parent[DS4_CHAINSTORE_ID_BYTES];    /* zeros: the chain begins here */
    uint8_t sent_id[DS4_CHAINSTORE_ID_BYTES];   /* a tail's second state, when sent_end != 0 */
    char *path;
    bool tail;
    uint32_t start, end;        /* positions [start, end) */
    uint32_t text_len;          /* of the history's text up to end */
    uint32_t sent_end;          /* a tail's state where the client's text ended; 0: none */
    uint32_t sent_text_len;
    uint64_t state_bytes;
    uint64_t file_size;
    uint64_t last_used;
} ds4_chainstore_node;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    int min_tokens;
    int segment_tokens;
    ds4_chainstore_node *node;
    int len, cap;
    uint8_t (*pin)[DS4_CHAINSTORE_ID_BYTES];    /* one per slot: the segment its live chain ends with */
    int n_pins;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg);
} ds4_chainstore;

typedef struct {
    int tokens;                 /* the position resumed; 0: nothing */
    uint32_t key_len;           /* bytes of the request's text the state stands for */
    bool own;                   /* a tail was resumed: it is the caller's to rewrite (tail_path) */
    char *tail_path;
    uint8_t parent[DS4_CHAINSTORE_ID_BYTES];    /* the sealed segment the resumed history ends with */
    uint32_t sealed_end;        /* its end; 0 with no parent */
    int segments;               /* files read */
    double load_ms;
} ds4_chainstore_load_result;

bool ds4_chainstore_open(ds4_chainstore *cs, const char *dir, uint64_t budget_mb, int min_tokens,
                         int n_slots, const char *log_name,
                         void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg), void *log_ud);
void ds4_chainstore_close(ds4_chainstore *cs);

/* Seal the live history's positions [sealed_end, live) as a segment under
 * `parent` (NULL or zeros: the chain's first).  The live state must stand on
 * text a client sent.  The id written to id_out names it; a segment already
 * there is left as it is. */
bool ds4_chainstore_seal(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                         const uint8_t *parent, uint32_t sealed_end,
                         const ds4_kvstore_trailer_hooks *hooks,
                         uint8_t id_out[DS4_CHAINSTORE_ID_BYTES], char *err, size_t err_len);

/* Write the slot's tail: positions [sealed_end, live), the live state and,
 * when sent_len lies between, the saved state there.  tail_path is the tail
 * this slot wrote or resumed before (replaced), or NULL.  Returns the path
 * (to free), NULL when nothing was written. */
char *ds4_chainstore_store_tail(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                                const uint8_t *parent, uint32_t sealed_end, int sent_len,
                                const char *tail_path, const char *reason,
                                const ds4_kvstore_trailer_hooks *hooks, char *err, size_t err_len);

/* Resume the longest state whose text begins prompt_text; a tail another
 * slot holds (held_tails, NULL-terminated, may be NULL) is read, not owned. */
int ds4_chainstore_load(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                        const char *prompt_text, const char *const *held_tails,
                        const ds4_kvstore_trailer_hooks *hooks,
                        ds4_chainstore_load_result *result);
void ds4_chainstore_load_result_free(ds4_chainstore_load_result *result);

/* A slot whose history went back to `upto` positions (an edit resumed from
 * a saved state) no longer stands on the segments past that: the deepest
 * segment of its chain that ends at or before `upto` is where it goes on
 * from, zeros and 0 when there is none.  The segments left behind stay, a
 * branch that may come back. */
void ds4_chainstore_ancestor_at(const ds4_chainstore *cs, const uint8_t *id, uint32_t upto,
                                uint8_t id_out[DS4_CHAINSTORE_ID_BYTES], uint32_t *end_out);

/* The chain a slot's live conversation ends with stays (zeros: none). */
void ds4_chainstore_pin(ds4_chainstore *cs, int slot, const uint8_t *id);
/* Make room for extra_bytes: unpinned leaves, least recently used first. */
void ds4_chainstore_evict(ds4_chainstore *cs, uint64_t extra_bytes);
/* Remove the tail of a slot whose history was sealed past it: every row it
 * holds is in a segment now. */
void ds4_chainstore_drop_tail(ds4_chainstore *cs, const char *tail_path);

/* Mark the chain prompt_text would resume as in use now. */
void ds4_chainstore_touch(ds4_chainstore *cs, ds4_engine *engine, ds4_session *session,
                          const char *prompt_text);
/* Every file's trailer, to hooks->load. */
void ds4_chainstore_load_trailers(ds4_chainstore *cs, const ds4_kvstore_trailer_hooks *hooks);

uint64_t ds4_chainstore_bytes(const ds4_chainstore *cs);

#endif
