#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "hallar/hallar.h"
#include "hallar/hallar.hpp"

struct hallar_model {
  std::shared_ptr<const hallar::Model> ptr;
};

struct hallar_engine {
  hallar::Engine eng;
};

namespace {

thread_local std::string tl_err;

template <typename F>
int guard(F&& fn) {
  try {
    fn();
    tl_err.clear();
    return 0;
  } catch (const std::exception& e) {
    tl_err = e.what();
  } catch (...) {
    tl_err = "hallar: unknown error";
  }
  return -1;
}

bool nulls(const void* a, const void* b = reinterpret_cast<const void*>(1)) {
  if (a && b) return false;
  tl_err = "hallar: null argument";
  return true;
}

char* dup(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (!p) throw std::bad_alloc();
  std::memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

}

extern "C" {

const char* hallar_version(void) { return "0.1.0"; }

const char* hallar_last_error(void) { return tl_err.c_str(); }

hallar_model* hallar_model_open(const char* path) {
  if (nulls(path)) return nullptr;
  hallar_model* out = nullptr;
  guard([&] { out = new hallar_model{hallar::Model::load(path)}; });
  return out;
}

void hallar_model_free(hallar_model* mdl) { delete mdl; }

size_t hallar_model_dim(const hallar_model* mdl) { return mdl ? mdl->ptr->dim() : 0; }

size_t hallar_model_vocab(const hallar_model* mdl) { return mdl ? mdl->ptr->vocab() : 0; }

int hallar_embed(const hallar_model* mdl, const char* text, float* out) {
  if (nulls(mdl, text) || nulls(out)) return -1;
  return guard([&] { mdl->ptr->embed_into(text, out); });
}

int hallar_tokenize(const hallar_model* mdl, const char* text, int32_t* out, size_t cap, size_t* len) {
  if (nulls(mdl, text) || nulls(len)) return -1;
  return guard([&] {
    const auto ids = mdl->ptr->tokenize(text);
    *len = ids.size();
    if (out) std::memcpy(out, ids.data(), std::min(cap, ids.size()) * sizeof(int32_t));
  });
}

hallar_engine* hallar_engine_new(const hallar_model* mdl, size_t m, size_t ef_build) {
  if (nulls(mdl)) return nullptr;
  hallar_engine* out = nullptr;
  guard([&] { out = new hallar_engine{hallar::Engine(mdl->ptr, m ? m : 16, ef_build ? ef_build : 200)}; });
  return out;
}

hallar_engine* hallar_engine_load(const hallar_model* mdl, const char* dir) {
  if (nulls(mdl, dir)) return nullptr;
  hallar_engine* out = nullptr;
  guard([&] { out = new hallar_engine{hallar::Engine::load(mdl->ptr, dir)}; });
  return out;
}

void hallar_engine_free(hallar_engine* eng) { delete eng; }

int hallar_engine_add(hallar_engine* eng, const char* id, const char* text) {
  if (nulls(eng, id) || nulls(text)) return -1;
  return guard([&] { eng->eng.add(id, text); });
}

int hallar_engine_search(const hallar_engine* eng, const char* query, size_t k, size_t ef, hallar_hits* out) {
  return hallar_engine_search_mode(eng, query, k, ef, HALLAR_HYBRID, out);
}

int hallar_engine_search_mode(const hallar_engine* eng, const char* query, size_t k, size_t ef, int mode,
                              hallar_hits* out) {
  if (nulls(eng, query) || nulls(out)) return -1;
  out->hits = nullptr;
  out->len = 0;
  if (mode < HALLAR_HYBRID || mode > HALLAR_KEYWORD) {
    tl_err = "hallar: unknown search mode";
    return -1;
  }
  return guard([&] {
    const auto hits = eng->eng.search(query, k, ef ? ef : 64, static_cast<hallar::Mode>(mode));
    if (hits.empty()) return;
    auto* arr = static_cast<hallar_hit*>(std::calloc(hits.size(), sizeof(hallar_hit)));
    if (!arr) throw std::bad_alloc();
    out->hits = arr;
    for (std::size_t i = 0; i < hits.size(); ++i) {
      arr[i].id = dup(hits[i].id);
      arr[i].text = dup(hits[i].text);
      arr[i].score = hits[i].score;
      out->len = i + 1;
    }
  });
}

int hallar_engine_set_weight(hallar_engine* eng, float vec_weight) {
  if (nulls(eng)) return -1;
  return guard([&] { eng->eng.set_weight(vec_weight); });
}

int hallar_engine_save(const hallar_engine* eng, const char* dir) {
  if (nulls(eng, dir)) return -1;
  return guard([&] { eng->eng.save(dir); });
}

size_t hallar_engine_size(const hallar_engine* eng) { return eng ? eng->eng.size() : 0; }

int hallar_engine_has(const hallar_engine* eng, const char* id) {
  if (!eng || !id) return 0;
  return eng->eng.has(id) ? 1 : 0;
}

void hallar_hits_free(hallar_hits* hits) {
  if (!hits || !hits->hits) return;
  for (std::size_t i = 0; i < hits->len; ++i) {
    std::free(const_cast<char*>(hits->hits[i].id));
    std::free(const_cast<char*>(hits->hits[i].text));
  }
  std::free(hits->hits);
  hits->hits = nullptr;
  hits->len = 0;
}

}
