#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "hallar/hallar.h"
#include "hallar/hallar.hpp"

using hallar::Engine;
using hallar::Model;
using hallar::Quant;
using hallar::Spec;

namespace {

int g_fail = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_fail;                                                     \
    }                                                               \
  } while (0)

const std::vector<std::string> kVocab = {"[PAD]", "[UNK]", "hola", "mundo", "##s", "un", "##aff", "##able",
                                         "casa", "!", "cafe", "perro", "gato", "##ito", "数", "?", "el", "la"};

std::shared_ptr<const Model> tiny(Quant q = Quant::F32, std::uint32_t flags = hallar::kLower | hallar::kStrip |
                                                                               hallar::kNorm | hallar::kCjk |
                                                                               hallar::kClean) {
  Spec sp;
  sp.dim = 4;
  sp.unk = 1;
  sp.flags = flags;
  sp.quant = q;
  std::vector<float> w(kVocab.size() * 4, 0.0f);
  for (std::size_t i = 0; i < kVocab.size(); ++i) {
    w[i * 4 + i % 4] = 1.0f + static_cast<float>(i) / 10.0f;
    w[i * 4 + (i + 1) % 4] = 0.25f;
  }
  return Model::build(sp, kVocab, w);
}

std::vector<std::string> toks(const Model& m, const std::string& s) {
  std::vector<std::string> out;
  for (auto id : m.tokenize(s)) out.push_back(m.token(id));
  return out;
}

bool same(const std::vector<std::string>& a, const std::vector<std::string>& b) { return a == b; }

std::string tmp(const std::string& name) {
  auto p = std::filesystem::temp_directory_path() / ("hallar_test_" + name);
  std::filesystem::remove_all(p);
  return p.string();
}

void test_normalize() {
  auto m = tiny();
  CHECK(m->normalize("Café  ÁRBOL\tñandú") == "cafe  arbol nandu");
  CHECK(m->normalize(std::string("a\0b", 3)) == "ab");
  CHECK(m->normalize("x\x01y z") == "xy z");
  CHECK(m->normalize("数学") == " 数  学 ");
  CHECK(m->normalize("ΣΑΣ İ") == "σασ i");
  CHECK(m->normalize("\xff\xfe ok") == " ok");
  auto cased = tiny(Quant::F32, hallar::kClean | hallar::kCjk);
  CHECK(cased->normalize("Café") == "Café");
}

void test_tokenize() {
  auto m = tiny();
  CHECK(same(m->pretokenize("hola, mundo!"), {"hola", ",", "mundo", "!"}));
  CHECK(same(toks(*m, "Hola mundos!"), {"hola", "mundo", "##s", "!"}));
  CHECK(same(toks(*m, "unaffable"), {"un", "##aff", "##able"}));
  CHECK(same(toks(*m, "perrito gatito"), {"perro", "[UNK]", "gato", "##ito"}) == false);
  CHECK(same(toks(*m, "gatito café?"), {"gato", "##ito", "cafe", "?"}) == false);
  CHECK(same(toks(*m, "xyz hola"), {"[UNK]", "hola"}));
  CHECK(same(toks(*m, "数"), {"数"}));
  CHECK(toks(*m, std::string(101, 'a')) == std::vector<std::string>{"[UNK]"});
  Spec sp = m->spec();
  CHECK(sp.max_tokens == 512);
}

void test_wordpiece_details() {
  auto m = tiny();
  CHECK(same(toks(*m, "gatoito"), {"gato", "##ito"}));
  CHECK(same(toks(*m, "holas"), {"hola", "##s"}));
  CHECK(same(toks(*m, "gatox"), {"[UNK]"}));
}

void test_truncation() {
  Spec sp;
  sp.dim = 2;
  sp.unk = 1;
  sp.max_tokens = 3;
  sp.max_chars = 11;
  auto m = Model::build(sp, {"[PAD]", "[UNK]", "a", "b"}, {0, 0, 0, 0, 1, 0, 0, 1});
  CHECK(m->tokenize("a b a b a").size() == 3);
  CHECK(m->normalize("abcdefghijklmnop").size() == 11);
}

void test_embed() {
  auto m = tiny();
  auto v = m->embed("hola mundo");
  double n = 0;
  for (float x : v) n += x * x;
  CHECK(std::fabs(n - 1.0) < 1e-5);
  auto z = m->embed("zzz qqq");
  for (float x : z) CHECK(x == 0.0f);
  auto a = m->embed("hola xyz");
  auto b = m->embed("hola");
  for (int i = 0; i < 4; ++i) CHECK(std::fabs(a[i] - b[i]) < 1e-6f);

  auto q = tiny(Quant::I8);
  auto vq = q->embed("hola mundo gato");
  auto vf = m->embed("hola mundo gato");
  double dot = 0;
  for (int i = 0; i < 4; ++i) dot += vq[i] * vf[i];
  CHECK(dot > 0.9999);
}

void test_model_io() {
  for (Quant q : {Quant::F32, Quant::I8}) {
    auto m = tiny(q);
    const std::string p = tmp("model.hallar");
    m->save(p);
    auto back = Model::load(p);
    CHECK(back->vocab() == m->vocab() && back->dim() == 4 && back->hash() == m->hash());
    CHECK(back->spec().quant == q && back->spec().flags == m->spec().flags);
    auto a = m->embed("el perro y la casa");
    auto b = back->embed("el perro y la casa");
    for (int i = 0; i < 4; ++i) CHECK(a[i] == b[i]);
    std::filesystem::resize_file(p, std::filesystem::file_size(p) / 2);
    bool threw = false;
    try {
      Model::load(p);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
    std::filesystem::remove(p);
  }
  bool threw = false;
  try {
    Model::build(Spec{}, {"a"}, {});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void test_engine() {
  auto m = tiny();
  Engine eng(m);
  eng.add("d1", "hola mundo");
  eng.add("d2", "el perro y el gato");
  eng.add("d3", "la casa del cafe");
  CHECK(eng.size() == 3 && eng.has("d2") && !eng.has("nope"));
  auto hits = eng.search("gato perro", 2);
  CHECK(hits.size() == 2 && hits[0].id == "d2" && hits[0].text == "el perro y el gato");
  CHECK(hits[0].score > hits[1].score && hits[0].score <= 1.0001f);
  bool threw = false;
  try {
    eng.add("d1", "otra vez");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  const std::string dir = tmp("engine");
  eng.save(dir);
  Engine back = Engine::load(m, dir);
  CHECK(back.size() == 3);
  for (const char* q : {"cafe", "gato perro", "hola"}) {
    auto a = eng.search(q, 3), b = back.search(q, 3);
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) CHECK(a[i].id == b[i].id && a[i].score == b[i].score);
  }
  CHECK(Engine::load(tiny(Quant::I8), dir).size() == 3);
  threw = false;
  try {
    Engine::load(tiny(Quant::F32, hallar::kClean), dir);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  Spec sp;
  sp.dim = 4;
  sp.unk = 1;
  auto other = Model::build(sp, {"x", "[UNK]"}, std::vector<float>(8, 1.0f));
  threw = false;
  try {
    Engine::load(other, dir);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("different model") != std::string::npos;
  }
  CHECK(threw);
  std::filesystem::remove_all(dir);
  CHECK(Engine(m).search("hola").empty());
}

void test_hybrid() {
  CHECK((hallar::terms("¡Hola, MUNDO! Árbol-de 数学") == std::vector<std::string>{"hola", "mundo", "arbol", "de", "数", "学"}));
  auto m = tiny();
  Engine eng(m);
  eng.add("a", "el perro come croquetas");
  eng.add("b", "la casa del cafe tiene un gato");
  eng.add("c", "receta: croquetas de jamón");
  auto kw = eng.search("croquetas jamón", 3, 64, hallar::Mode::Keyword);
  CHECK(kw.size() == 2 && kw[0].id == "c" && kw[1].id == "a");
  auto none = eng.search("zzz", 3, 64, hallar::Mode::Keyword);
  CHECK(none.empty());
  auto vec = eng.search("zzz qqq", 3, 64, hallar::Mode::Vector);
  CHECK(vec.empty());
  auto hy = eng.search("croquetas jamón", 3);
  CHECK(!hy.empty() && hy[0].id == "c");
  for (std::size_t i = 1; i < hy.size(); ++i) CHECK(hy[i - 1].score >= hy[i].score);
  const std::string dir = tmp("hybrid");
  eng.save(dir);
  Engine back = Engine::load(m, dir);
  auto again = back.search("croquetas jamón", 3, 64, hallar::Mode::Keyword);
  CHECK(again.size() == kw.size() && again[0].id == "c" && again[0].score == kw[0].score);
  std::filesystem::remove_all(dir);
  hallar_hits hits;
  CHECK(hallar_engine_search_mode(nullptr, "x", 1, 0, 0, &hits) == -1);
}

void test_concurrency() {
  auto m = tiny();
  Engine eng(m);
  eng.add("seed", "hola");
  std::atomic<bool> stop{false};
  std::atomic<int> found{0};
  std::thread reader([&] {
    while (!stop) found += static_cast<int>(eng.search("perro gato", 3).size());
  });
  for (int i = 0; i < 400; ++i) eng.add("doc" + std::to_string(i), i % 2 ? "el perro" : "la casa del gato");
  stop = true;
  reader.join();
  CHECK(eng.size() == 401 && found > 0);
}

void test_capi() {
  auto m = tiny();
  const std::string p = tmp("capi.hallar");
  m->save(p);
  hallar_model* hm = hallar_model_open(p.c_str());
  CHECK(hm && hallar_model_dim(hm) == 4 && hallar_model_vocab(hm) == kVocab.size());
  CHECK(hallar_model_open("/nope/x.hallar") == nullptr && std::string(hallar_last_error()).find("cannot open") != std::string::npos);
  float v[4];
  CHECK(hallar_embed(hm, "hola", v) == 0);
  std::int32_t ids[8];
  std::size_t n = 0;
  CHECK(hallar_tokenize(hm, "hola mundos", ids, 8, &n) == 0 && n == 3 && ids[0] == 2 && ids[2] == 4);
  CHECK(hallar_tokenize(hm, "hola mundos", nullptr, 0, &n) == 0 && n == 3);
  hallar_engine* he = hallar_engine_new(hm, 0, 0);
  CHECK(hallar_engine_add(he, "a", "hola mundo") == 0);
  CHECK(hallar_engine_add(he, "b", "el gato") == 0);
  CHECK(hallar_engine_add(he, "a", "dup") == -1 && std::string(hallar_last_error()).find("duplicate") != std::string::npos);
  hallar_hits hits;
  CHECK(hallar_engine_search_mode(he, "gato", 5, 0, HALLAR_VECTOR, &hits) == 0 && hits.len == 2 && std::string(hits.hits[0].id) == "b");
  hallar_hits_free(&hits);
  CHECK(hallar_engine_search_mode(he, "gato", 5, 0, 9, &hits) == -1 && std::string(hallar_last_error()).find("mode") != std::string::npos);
  CHECK(hallar_engine_search(he, "gato", 5, 0, &hits) == 0 && hits.len >= 1 && std::string(hits.hits[0].id) == "b");
  hallar_hits_free(&hits);
  CHECK(hits.hits == nullptr && hits.len == 0);
  const std::string dir = tmp("capi_dir");
  CHECK(hallar_engine_save(he, dir.c_str()) == 0);
  hallar_engine* back = hallar_engine_load(hm, dir.c_str());
  CHECK(back && hallar_engine_size(back) == 2 && hallar_engine_has(back, "a") == 1 && hallar_engine_has(back, "z") == 0);
  CHECK(hallar_engine_add(nullptr, "x", "y") == -1 && std::string(hallar_last_error()) == "hallar: null argument");
  CHECK(hallar_engine_size(nullptr) == 0 && hallar_model_dim(nullptr) == 0);
  hallar_engine_free(back);
  hallar_engine_free(he);
  hallar_model_free(hm);
  hallar_hits_free(nullptr);
  std::filesystem::remove_all(dir);
  std::filesystem::remove(p);
  CHECK(std::string(hallar_version()) == "0.1.0" && std::string(hallar::unicode_version()).size() > 0);
}

}

int main() {
  struct T {
    const char* name;
    std::function<void()> fn;
  };
  const std::vector<T> all = {{"normalize", test_normalize}, {"tokenize", test_tokenize},
                              {"wordpiece_details", test_wordpiece_details}, {"truncation", test_truncation},
                              {"embed", test_embed}, {"model_io", test_model_io}, {"engine", test_engine},
                              {"hybrid", test_hybrid}, {"concurrency", test_concurrency}, {"capi", test_capi}};
  for (const T& t : all) {
    const int before = g_fail;
    std::printf("%s\n", t.name);
    t.fn();
    std::printf("  %s\n", g_fail == before ? "ok" : "FAILED");
  }
  std::printf("%d failure(s)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
