#ifndef DS4_KVSTORE_H
#define DS4_KVSTORE_H

#include "ds4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_KVSTORE_FIXED_HEADER 48u
#define DS4_KVSTORE_DEFAULT_MB 4096

#define DS4_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define DS4_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define DS4_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define DS4_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    DS4_KVSTORE_REASON_UNKNOWN   = 0,
    DS4_KVSTORE_REASON_COLD      = 1,
    DS4_KVSTORE_REASON_CONTINUED = 2,
    DS4_KVSTORE_REASON_EVICT     = 3,
    DS4_KVSTORE_REASON_SHUTDOWN  = 4,
    DS4_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    DS4_KVSTORE_REASON_AGENT_SESSION = 6,
} ds4_kvstore_reason;

typedef enum {
    DS4_KVSTORE_LOG_DEFAULT,
    DS4_KVSTORE_LOG_KVCACHE,
    DS4_KVSTORE_LOG_WARNING,
} ds4_kvstore_log_type;

/* One point of a file's history a session can resume from: its position,
 * and the prompt bytes it stands for (the rendered text of the history up
 * to there, or a visible-transcript key of its own kept in the file). */
typedef struct {
    uint32_t position;
    uint32_t key_len;
    char sha[41];          /* of the key bytes */
    uint64_t offset;       /* the state blob */
    uint64_t bytes;
    uint64_t key_offset;   /* the key bytes when they are not the text's prefix; 0 otherwise */
} ds4_kvstore_state;

/* A checkpoint file holds one conversation: the blocks of its history
 * (what every position contributes, written once and appended to) and a
 * tail with the tokens, the rendered text, the states the session can
 * resume from, and the protocol trailers. */
typedef struct {
    char sha[41];          /* the file name */
    char *path;
    uint8_t quant_bits;
    uint8_t model_id;
    uint8_t reason;        /* of the last store */
    uint8_t ext_flags;
    uint32_t tokens;       /* the history's length */
    uint32_t hits;
    uint32_t ctx_size;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t file_size;
    uint64_t tail_offset;
    uint32_t blocks_end;   /* positions the blocks cover */
    uint32_t block_positions;
    uint32_t text_bytes;
    uint32_t tail_tokens;  /* tokens listed in the tail (none in a text-only file) */
    uint64_t tokens_offset;
    uint64_t text_offset;
    uint64_t trailer_offset;
    ds4_kvstore_state *state;
    uint32_t n_states;
} ds4_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} ds4_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    ds4_kvstore_options opt;
    int continued_last_store_tokens;
    ds4_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg);
} ds4_kvstore;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} ds4_kvstore_trailer_hooks;

/* What to store: the session's first store_len tokens (they must be its
 * whole live history), under the store's reason.  A key override keys the
 * live state by a visible transcript instead of the rendered text.  The
 * store extends the file at extend_path when the session grew out of it,
 * writes to path when one is given (extending it when it can), and
 * otherwise writes a new file named by the text; created_at 0 means now. */
typedef struct {
    const ds4_tokens *tokens;
    int store_len;
    const char *reason;
    const char *key_override;
    uint8_t key_ext;
    const char *key_kind;
    const ds4_kvstore_trailer_hooks *hooks;
    const char *extend_path;
    const char *path;
    uint64_t created_at;
} ds4_kvstore_store_request;

typedef struct {
    int tokens;            /* the position resumed */
    uint32_t key_len;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} ds4_kvstore_load_result;

ds4_kvstore_options ds4_kvstore_default_options(void);
uint8_t ds4_kvstore_reason_code(const char *reason);
const char *ds4_kvstore_key_kind(uint8_t ext_flags);

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud);
void ds4_kvstore_close(ds4_kvstore *kc);
void ds4_kvstore_clear(ds4_kvstore *kc);
void ds4_kvstore_entry_free(ds4_kvstore_entry *e);

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len);
bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n);
void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out);

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens);
int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens);
void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens);
int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens);
void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc, uint64_t file_bytes,
                                uint64_t *required_bytes_out);
double ds4_kvstore_entry_eviction_score(const ds4_kvstore_entry *e);
/* Make room for extra_bytes under the budget, least recently used first. */
void ds4_kvstore_evict(ds4_kvstore *kc, uint64_t extra_bytes);
/* The file and state whose key is the longest prefix of prompt_text. */
int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size,
                                 uint32_t *state_out);

/* Store; the path written (to free) or NULL.  kc may be NULL: no budget,
 * no eviction, no log. */
char *ds4_kvstore_store(ds4_kvstore *kc, ds4_engine *engine, ds4_session *session,
                        const ds4_kvstore_store_request *req, char *err, size_t err_len);
bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       const char *extend_path,
                                       char **path_out,
                                       char *err,
                                       size_t err_len);
/* A file with the tokens and text of a history but no state: the agent's
 * stripped sessions, rebuilt from text when opened. */
bool ds4_kvstore_write_text_only(const char *path, uint8_t model_id, uint8_t quant_bits,
                                 uint8_t reason, uint8_t ext_flags, uint32_t tokens,
                                 uint32_t ctx_size, uint64_t created_at, const char *text,
                                 const ds4_kvstore_trailer_hooks *hooks,
                                 char *err, size_t err_len);

/* Resume a file's state (by index in its index) into the session; the
 * position resumed, 0 on failure. */
int ds4_kvstore_load(ds4_engine *engine, ds4_session *session, FILE *fp,
                     const ds4_kvstore_entry *e, uint32_t state_index,
                     const ds4_kvstore_trailer_hooks *hooks,
                     char *err, size_t err_len);
int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result);

/* The file's header and index; e->path and e->sha are the caller's. */
bool ds4_kvstore_read_index(FILE *fp, ds4_kvstore_entry *e);
bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out);
char *ds4_kvstore_read_text(FILE *fp, const ds4_kvstore_entry *e);
/* Record a use of the file: its hit count and when (0: now). */
bool ds4_kvstore_touch_file(const char *path, uint32_t hits, uint64_t used_at);
bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]);
void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *ds4_kvstore_path_join(const char *dir, const char *name);
char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]);
void ds4_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t ds4_kvstore_le_get32(const uint8_t *p);

#endif
