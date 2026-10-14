#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

#include <algorithm>
#include <cmath>

#include "hallar/hallar.hpp"
#include "text.hpp"
#include "vecdex/index.hpp"

namespace hallar {

namespace {

constexpr std::uint32_t kDocs = 0x534F4448;

template <typename T>
void put(std::ofstream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void put(std::ofstream& out, const std::string& s) {
  put(out, static_cast<std::uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

template <typename T>
T get(std::ifstream& in) {
  T v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("hallar: truncated docs file");
  return v;
}

std::string str(std::ifstream& in, std::uint64_t cap) {
  const auto n = get<std::uint32_t>(in);
  if (n > cap) throw std::runtime_error("hallar: corrupt docs file");
  std::string s(n, '\0');
  in.read(s.data(), n);
  if (!in) throw std::runtime_error("hallar: truncated docs file");
  return s;
}

}

std::vector<std::string> terms(std::string_view raw) {
  std::vector<std::string> out;
  std::string cur;
  std::u32string tmp, low;
  auto flush = [&] {
    if (!cur.empty()) out.push_back(std::move(cur)), cur.clear();
  };
  for (char32_t cp : text::decode(raw)) {
    if (cp == 0 || cp == 0xFFFD || text::control(cp) || text::space(cp) || text::punct(cp)) {
      flush();
      continue;
    }
    if (text::cjk(cp)) {
      flush();
      text::encode(cp, cur);
      flush();
      continue;
    }
    tmp.clear();
    text::nfd(cp, tmp);
    for (char32_t d : tmp) {
      if (text::mark(d)) continue;
      low.clear();
      text::lower(d, low);
      for (char32_t c : low) text::encode(c, cur);
    }
  }
  flush();
  return out;
}

Engine::Engine(std::shared_ptr<const Model> mdl, std::size_t m, std::size_t ef_build)
    : mdl_(std::move(mdl)), mu_(std::make_unique<std::shared_mutex>()) {
  if (!mdl_) throw std::invalid_argument("hallar: null model");
  idx_ = std::make_unique<vecdex::Index>(mdl_->dim(), vecdex::Metric::Cosine, m, ef_build);
}

Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

bool Engine::has(const std::string& id) const {
  std::shared_lock lk(*mu_);
  return slot_.count(id) > 0;
}

void Engine::set_weight(float w) {
  if (!(w >= 0.0f && w <= 1.0f)) throw std::invalid_argument("hallar: weight must be between 0 and 1");
  std::unique_lock lk(*mu_);
  wvec_ = w;
}

float Engine::weight() const {
  std::shared_lock lk(*mu_);
  return wvec_;
}

std::size_t Engine::size() const {
  std::shared_lock lk(*mu_);
  return ids_.size();
}

void Engine::add(const std::string& id, const std::string& txt) {
  const std::vector<float> vec = mdl_->embed(txt);
  std::unique_lock lk(*mu_);
  if (slot_.count(id)) throw std::invalid_argument("hallar: duplicate id " + id);
  const auto pos = static_cast<std::uint64_t>(ids_.size());
  idx_->add(vec.data(), pos);
  ids_.push_back(id);
  texts_.push_back(txt);
  slot_.emplace(id, pos);
  index_terms(static_cast<std::uint32_t>(pos), txt);
}

void Engine::index_terms(std::uint32_t doc, const std::string& txt) {
  std::unordered_map<std::string, std::uint32_t> tf;
  const auto words = terms(txt);
  for (const auto& w : words) ++tf[w];
  for (auto& [w, n] : tf) post_[w].emplace_back(doc, n);
  lens_.push_back(static_cast<std::uint32_t>(words.size()));
  total_ += words.size();
}

std::vector<std::pair<std::uint32_t, float>> Engine::vector_rank(std::string_view q, std::size_t n,
                                                                 std::size_t ef) const {
  const std::vector<float> vec = mdl_->embed(q);
  std::vector<std::pair<std::uint32_t, float>> out;
  bool any = false;
  for (float x : vec) any = any || x != 0.0f;
  if (!any) return out;
  for (const vecdex::Hit& h : idx_->search(vec.data(), n, std::max(ef, n))) {
    out.emplace_back(static_cast<std::uint32_t>(h.id), 1.0f - h.dist);
  }
  return out;
}

std::vector<std::pair<std::uint32_t, float>> Engine::keyword_rank(std::string_view q, std::size_t n) const {
  std::vector<std::pair<std::uint32_t, float>> out;
  const std::size_t docs = lens_.size();
  if (docs == 0) return out;
  const double avg = static_cast<double>(total_) / static_cast<double>(docs);
  constexpr double k1 = 1.2, b = 0.75;
  std::unordered_map<std::uint32_t, double> acc;
  auto qs = terms(q);
  std::sort(qs.begin(), qs.end());
  qs.erase(std::unique(qs.begin(), qs.end()), qs.end());
  for (const auto& w : qs) {
    const auto it = post_.find(w);
    if (it == post_.end()) continue;
    const double df = static_cast<double>(it->second.size());
    const double idf = std::log(1.0 + (static_cast<double>(docs) - df + 0.5) / (df + 0.5));
    for (const auto& [doc, f] : it->second) {
      const double tf = f;
      acc[doc] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens_[doc] / (avg > 0 ? avg : 1.0)));
    }
  }
  out.reserve(acc.size());
  for (const auto& [doc, sc] : acc) out.emplace_back(doc, static_cast<float>(sc));
  const std::size_t keep = std::min(n, out.size());
  std::partial_sort(out.begin(), out.begin() + static_cast<long>(keep), out.end(), [](const auto& x, const auto& y) {
    return x.second != y.second ? x.second > y.second : x.first < y.first;
  });
  out.resize(keep);
  return out;
}

std::vector<Hit> Engine::search(std::string_view qry, std::size_t k, std::size_t ef, Mode mode) const {
  std::shared_lock lk(*mu_);
  std::vector<Hit> out;
  if (k == 0 || ids_.empty()) return out;
  std::vector<std::pair<std::uint32_t, float>> ranked;
  if (mode == Mode::Vector) {
    ranked = vector_rank(qry, k, ef);
  } else if (mode == Mode::Keyword) {
    ranked = keyword_rank(qry, k);
  } else {
    const std::size_t pool = std::max<std::size_t>(k * 4, 50);
    constexpr float kRrf = 60.0f;
    std::unordered_map<std::uint32_t, float> fused;
    const auto vr = vector_rank(qry, pool, ef);
    const auto kr = keyword_rank(qry, pool);
    for (std::size_t i = 0; i < vr.size(); ++i) fused[vr[i].first] += wvec_ / (kRrf + static_cast<float>(i + 1));
    for (std::size_t i = 0; i < kr.size(); ++i) {
      fused[kr[i].first] += (1.0f - wvec_) / (kRrf + static_cast<float>(i + 1));
    }
    ranked.assign(fused.begin(), fused.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y) {
      return x.second != y.second ? x.second > y.second : x.first < y.first;
    });
    if (ranked.size() > k) ranked.resize(k);
  }
  out.reserve(ranked.size());
  for (const auto& [doc, sc] : ranked) out.push_back({ids_[doc], sc, texts_[doc]});
  return out;
}

void Engine::save(const std::string& dir) const {
  std::shared_lock lk(*mu_);
  std::filesystem::create_directories(dir);
  idx_->save(dir + "/index.vdx");
  std::ofstream out(dir + "/docs.bin", std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("hallar: cannot write " + dir);
  put(out, kDocs);
  put(out, mdl_->hash());
  put(out, static_cast<std::uint64_t>(ids_.size()));
  for (std::size_t i = 0; i < ids_.size(); ++i) {
    put(out, ids_[i]);
    put(out, texts_[i]);
  }
  if (!out) throw std::runtime_error("hallar: write failed for " + dir);
}

Engine Engine::load(std::shared_ptr<const Model> mdl, const std::string& dir) {
  Engine eng(std::move(mdl));
  const std::string docs = dir + "/docs.bin";
  std::ifstream in(docs, std::ios::binary);
  if (!in) throw std::runtime_error("hallar: cannot open " + docs);
  const auto bytes = static_cast<std::uint64_t>(std::filesystem::file_size(docs));
  if (get<std::uint32_t>(in) != kDocs) throw std::runtime_error("hallar: not a docs file");
  if (get<std::uint64_t>(in) != eng.mdl_->hash()) {
    throw std::runtime_error("hallar: index was built with a different model");
  }
  const auto n = get<std::uint64_t>(in);
  if (n > bytes) throw std::runtime_error("hallar: corrupt docs file");
  for (std::uint64_t i = 0; i < n; ++i) {
    std::string id = str(in, bytes);
    std::string txt = str(in, bytes);
    if (!eng.slot_.emplace(id, i).second) throw std::runtime_error("hallar: duplicate id in docs file");
    eng.index_terms(static_cast<std::uint32_t>(i), txt);
    eng.ids_.push_back(std::move(id));
    eng.texts_.push_back(std::move(txt));
  }
  *eng.idx_ = vecdex::Index::load(dir + "/index.vdx");
  if (eng.idx_->dim() != eng.mdl_->dim() || eng.idx_->size() != n) {
    throw std::runtime_error("hallar: index and docs do not match");
  }
  return eng;
}

}
