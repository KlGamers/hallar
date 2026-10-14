#ifndef HALLAR_H
#define HALLAR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define HALLAR_API __declspec(dllexport)
#else
#define HALLAR_API __attribute__((visibility("default")))
#endif

typedef struct hallar_model hallar_model;
typedef struct hallar_engine hallar_engine;

typedef struct {
  const char* id;
  const char* text;
  float score;
} hallar_hit;

typedef struct {
  hallar_hit* hits;
  size_t len;
} hallar_hits;

HALLAR_API const char* hallar_version(void);
HALLAR_API const char* hallar_last_error(void);

HALLAR_API hallar_model* hallar_model_open(const char* path);
HALLAR_API void hallar_model_free(hallar_model* mdl);
HALLAR_API size_t hallar_model_dim(const hallar_model* mdl);
HALLAR_API size_t hallar_model_vocab(const hallar_model* mdl);
HALLAR_API int hallar_embed(const hallar_model* mdl, const char* text, float* out);
HALLAR_API int hallar_tokenize(const hallar_model* mdl, const char* text, int32_t* out, size_t cap, size_t* len);

HALLAR_API hallar_engine* hallar_engine_new(const hallar_model* mdl, size_t m, size_t ef_build);
HALLAR_API hallar_engine* hallar_engine_load(const hallar_model* mdl, const char* dir);
HALLAR_API void hallar_engine_free(hallar_engine* eng);
HALLAR_API int hallar_engine_add(hallar_engine* eng, const char* id, const char* text);
#define HALLAR_HYBRID 0
#define HALLAR_VECTOR 1
#define HALLAR_KEYWORD 2

HALLAR_API int hallar_engine_search(const hallar_engine* eng, const char* query, size_t k, size_t ef, hallar_hits* out);
HALLAR_API int hallar_engine_search_mode(const hallar_engine* eng, const char* query, size_t k, size_t ef, int mode,
                                         hallar_hits* out);
HALLAR_API int hallar_engine_set_weight(hallar_engine* eng, float vec_weight);
HALLAR_API int hallar_engine_save(const hallar_engine* eng, const char* dir);
HALLAR_API size_t hallar_engine_size(const hallar_engine* eng);
HALLAR_API int hallar_engine_has(const hallar_engine* eng, const char* id);
HALLAR_API void hallar_hits_free(hallar_hits* hits);

#ifdef __cplusplus
}
#endif

#endif
