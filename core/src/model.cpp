#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "hallar/hallar.hpp"
#include "text.hpp"

namespace hallar {

namespace {

constexpr std::uint32_t kMagic = 0x524C4C48;
constexpr std::uint32_t kVersion = 2;

template <typename T>
void put(std::ofstream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void put(std::ofstream& out, const std::string& s) {
  put(out, static_cast<std::uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

template <typename T>
void put(std::ofstream& out, const std::vector<T>& v) {
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

struct Reader {
  std::ifstream in;
  std::uint64_t left;

  void raw(void* dst, std::uint64_t n) {
    if (n > left) throw std::runtime_error("hallar: truncated model file");
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("hallar: truncated model file");
    left -= n;
  }

  template <typename T>
  T get() {
    T v{};
    raw(&v, sizeof(T));
    return v;
  }

  std::string str() {
    const auto n = get<std::uint32_t>();
    std::string s(n, '\0');
    raw(s.data(), n);
    return s;
  }
};

std::uint64_t fnv(const std::vector<std::string>& vocab, const Spec& sp) {
  std::uint64_t h = 1469598103934665603ull ^ sp.dim ^ (std::uint64_t(sp.flags) << 32);
  auto mix = [&h](std::string_view s) {
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    h = (h ^ 0xFE) * 1099511628211ull;
  };
  mix(sp.prefix);
  mix(sp.meta);
  h ^= (std::uint64_t(sp.algo) << 8) ^ (std::uint64_t(sp.pre) << 16) ^ (std::uint64_t(sp.prepend) << 24);
  for (const Op& op : sp.ops) {
    h = (h ^ static_cast<std::uint64_t>(op.kind)) * 1099511628211ull;
    mix(op.a);
    mix(op.b);
    h = (h ^ op.arg) * 1099511628211ull;
  }
  for (const auto& tok : vocab) {
    for (unsigned char c : tok) h = (h ^ c) * 1099511628211ull;
    h = (h ^ 0xFF) * 1099511628211ull;
  }
  return h;
}

}

const char* unicode_version() { return text::version(); }

std::shared_ptr<const Model> Model::build(Spec spec, std::vector<std::string> vocab, std::vector<float> weights,
                                          std::vector<double> scores) {
  if (spec.dim == 0) throw std::invalid_argument("hallar: dim must be positive");
  if (vocab.empty()) throw std::invalid_argument("hallar: empty vocabulary");
  if (weights.size() != vocab.size() * spec.dim) throw std::invalid_argument("hallar: weights do not match vocab x dim");
  if (spec.unk >= static_cast<std::int32_t>(vocab.size())) throw std::invalid_argument("hallar: unk id out of range");
  if (spec.algo == Algo::Unigram && scores.size() != vocab.size()) {
    throw std::invalid_argument("hallar: unigram models need one score per token");
  }
  auto mdl = std::shared_ptr<Model>(new Model());
  mdl->scores_ = std::move(scores);
  mdl->spec_ = std::move(spec);
  mdl->vocab_ = std::move(vocab);
  const std::size_t dim = mdl->spec_.dim;
  if (mdl->spec_.quant == Quant::I8) {
    const std::size_t n = mdl->vocab_.size();
    mdl->scale_.resize(n);
    mdl->i8_.resize(n * dim);
    for (std::size_t r = 0; r < n; ++r) {
      float mx = 0.0f;
      for (std::size_t c = 0; c < dim; ++c) mx = std::max(mx, std::fabs(weights[r * dim + c]));
      const float sc = mx > 0.0f ? mx / 127.0f : 1.0f;
      mdl->scale_[r] = sc;
      for (std::size_t c = 0; c < dim; ++c) {
        const float q = std::round(weights[r * dim + c] / sc);
        mdl->i8_[r * dim + c] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
      }
    }
  } else {
    mdl->f32_ = std::move(weights);
  }
  mdl->index_vocab();
  return mdl;
}

void Model::index_vocab() {
  ids_.clear();
  edge_.clear();
  leaf_.clear();
  if (spec_.algo == Algo::WordPiece) {
    ids_.reserve(vocab_.size());
    for (std::size_t i = 0; i < vocab_.size(); ++i) ids_.emplace(vocab_[i], static_cast<std::int32_t>(i));
  } else {
    leaf_.push_back(-1);
    edge_.reserve(vocab_.size() * 4);
    double lo = 0.0;
    for (std::size_t i = 0; i < vocab_.size(); ++i) {
      if (static_cast<std::int32_t>(i) == spec_.unk || vocab_[i].empty()) continue;
      lo = std::min(lo, scores_[i]);
      std::uint32_t node = 0;
      for (unsigned char c : vocab_[i]) {
        const std::uint64_t key = (std::uint64_t(node) << 8) | c;
        const auto it = edge_.find(key);
        if (it == edge_.end()) {
          const auto nxt = static_cast<std::uint32_t>(leaf_.size());
          leaf_.push_back(-1);
          edge_.emplace(key, nxt);
          node = nxt;
        } else {
          node = it->second;
        }
      }
      if (leaf_[node] < 0) leaf_[node] = static_cast<std::int32_t>(i);
    }
    unk_score_ = lo - 10.0;
  }
  hash_ = fnv(vocab_, spec_);
}

void Model::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("hallar: cannot open " + path);
  put(out, kMagic);
  put(out, kVersion);
  put(out, spec_.dim);
  put(out, static_cast<std::uint32_t>(vocab_.size()));
  put(out, spec_.unk);
  put(out, spec_.flags);
  put(out, spec_.max_tokens);
  put(out, spec_.max_chars);
  put(out, spec_.max_word);
  put(out, static_cast<std::uint32_t>(spec_.quant));
  put(out, spec_.prefix);
  put(out, static_cast<std::uint32_t>(spec_.algo));
  put(out, static_cast<std::uint32_t>(spec_.pre));
  put(out, spec_.meta);
  put(out, spec_.prepend);
  put(out, static_cast<std::uint32_t>(spec_.ops.size()));
  for (const Op& op : spec_.ops) {
    put(out, static_cast<std::uint32_t>(op.kind));
    put(out, op.a);
    put(out, op.b);
    put(out, op.arg);
  }
  for (const auto& tok : vocab_) put(out, tok);
  if (spec_.algo == Algo::Unigram) put(out, scores_);
  if (spec_.quant == Quant::I8) {
    put(out, scale_);
    put(out, i8_);
  } else {
    put(out, f32_);
  }
  if (!out) throw std::runtime_error("hallar: write failed for " + path);
}

std::shared_ptr<const Model> Model::load(const std::string& path) {
  Reader rd{std::ifstream(path, std::ios::binary), 0};
  if (!rd.in) throw std::runtime_error("hallar: cannot open " + path);
  rd.in.seekg(0, std::ios::end);
  rd.left = static_cast<std::uint64_t>(rd.in.tellg());
  rd.in.seekg(0);
  if (rd.get<std::uint32_t>() != kMagic) throw std::runtime_error("hallar: not a model file");
  if (rd.get<std::uint32_t>() != kVersion) throw std::runtime_error("hallar: unsupported model version");
  auto mdl = std::shared_ptr<Model>(new Model());
  Spec& sp = mdl->spec_;
  sp.dim = rd.get<std::uint32_t>();
  const auto n = rd.get<std::uint32_t>();
  sp.unk = rd.get<std::int32_t>();
  sp.flags = rd.get<std::uint32_t>();
  sp.max_tokens = rd.get<std::uint32_t>();
  sp.max_chars = rd.get<std::uint32_t>();
  sp.max_word = rd.get<std::uint32_t>();
  const auto q = rd.get<std::uint32_t>();
  if (sp.dim == 0 || n == 0 || q > 1 || sp.unk >= static_cast<std::int32_t>(n)) {
    throw std::runtime_error("hallar: corrupt model file");
  }
  sp.quant = static_cast<Quant>(q);
  sp.prefix = rd.str();
  const auto algo = rd.get<std::uint32_t>();
  const auto pre = rd.get<std::uint32_t>();
  if (algo > 1 || pre > 1) throw std::runtime_error("hallar: corrupt model file");
  sp.algo = static_cast<Algo>(algo);
  sp.pre = static_cast<Pre>(pre);
  sp.meta = rd.str();
  sp.prepend = rd.get<std::uint32_t>();
  const auto nops = rd.get<std::uint32_t>();
  if (nops > 4096) throw std::runtime_error("hallar: corrupt model file");
  sp.ops.clear();
  for (std::uint32_t i = 0; i < nops; ++i) {
    Op op;
    const auto kind = rd.get<std::uint32_t>();
    if (kind > 5) throw std::runtime_error("hallar: corrupt model file");
    op.kind = static_cast<OpKind>(kind);
    op.a = rd.str();
    op.b = rd.str();
    op.arg = rd.get<std::uint32_t>();
    sp.ops.push_back(std::move(op));
  }
  const std::uint64_t need = (sp.quant == Quant::I8 ? std::uint64_t(n) * (4 + sp.dim) : std::uint64_t(n) * sp.dim * 4) +
                             (sp.algo == Algo::Unigram ? std::uint64_t(n) * 8 : 0);
  if (need > rd.left) throw std::runtime_error("hallar: truncated model file");
  mdl->vocab_.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) mdl->vocab_.push_back(rd.str());
  if (sp.algo == Algo::Unigram) {
    mdl->scores_.resize(n);
    rd.raw(mdl->scores_.data(), std::uint64_t(n) * 8);
  }
  if (sp.quant == Quant::I8) {
    mdl->scale_.resize(n);
    mdl->i8_.resize(std::size_t(n) * sp.dim);
    rd.raw(mdl->scale_.data(), mdl->scale_.size() * 4);
    rd.raw(mdl->i8_.data(), mdl->i8_.size());
  } else {
    mdl->f32_.resize(std::size_t(n) * sp.dim);
    rd.raw(mdl->f32_.data(), mdl->f32_.size() * 4);
  }
  mdl->index_vocab();
  return mdl;
}

std::string Model::bert(std::string_view in) const {
  const std::uint32_t fl = spec_.flags;
  std::u32string a;
  const std::u32string cps = text::decode(in);
  a.reserve(cps.size());
  for (char32_t cp : cps) {
    if (fl & kClean) {
      if (cp == 0 || cp == 0xFFFD || text::control(cp)) continue;
      if (text::space(cp)) cp = U' ';
    }
    if ((fl & kCjk) && text::cjk(cp)) {
      a.push_back(U' ');
      a.push_back(cp);
      a.push_back(U' ');
    } else {
      a.push_back(cp);
    }
  }
  if (fl & kStrip) {
    std::u32string b, tmp;
    b.reserve(a.size());
    for (char32_t cp : a) {
      tmp.clear();
      text::nfd(cp, tmp);
      for (char32_t d : tmp) {
        if (!text::mark(d)) b.push_back(d);
      }
    }
    a.swap(b);
  }
  if (fl & kLower) {
    std::u32string b;
    b.reserve(a.size());
    for (char32_t cp : a) text::lower(cp, b);
    a.swap(b);
  }
  return text::encode(a);
}

std::string Model::precompiled(const Op& op, std::string_view in) const {
  const std::string& blob = op.a;
  if (blob.size() < 4) return std::string(in);
  std::uint32_t tsz = 0;
  std::memcpy(&tsz, blob.data(), 4);
  if (tsz % 4 != 0 || 4 + std::size_t(tsz) > blob.size()) return std::string(in);
  const auto* units = reinterpret_cast<const unsigned char*>(blob.data() + 4);
  const std::size_t nunits = tsz / 4;
  const char* pool = blob.data() + 4 + tsz;
  const std::size_t psz = blob.size() - 4 - tsz;
  auto unit = [&](std::size_t i) {
    std::uint32_t u;
    std::memcpy(&u, units + i * 4, 4);
    return u;
  };
  auto offset = [](std::uint32_t u) { return (u >> 10) << ((u & (1u << 9)) >> 6); };
  auto lookup = [&](std::string_view chunk, std::string& out) {
    if (nunits == 0) return false;
    std::size_t pos = 0;
    std::uint32_t u = unit(0);
    pos ^= offset(u);
    for (unsigned char c : chunk) {
      pos ^= c;
      if (pos >= nunits) return false;
      u = unit(pos);
      if ((u & ((1u << 31) | 0xFF)) != c) return false;
      pos ^= offset(u);
      if (pos >= nunits) return false;
      if ((u >> 8) & 1) {
        const std::uint32_t val = unit(pos) & ((1u << 31) - 1);
        if (val >= psz) return false;
        const char* st = pool + val;
        out.append(st, ::strnlen(st, psz - val));
        return true;
      }
    }
    return false;
  };
  std::string out;
  out.reserve(in.size());
  const std::vector<std::size_t> cut = text::graphemes(in);
  for (std::size_t g = 0; g + 1 < cut.size(); ++g) {
    const std::string_view gr = in.substr(cut[g], cut[g + 1] - cut[g]);
    if (gr.size() < 6 && lookup(gr, out)) continue;
    for (std::size_t i = 0; i < gr.size();) {
      std::size_t len = 1;
      const auto c0 = static_cast<unsigned char>(gr[i]);
      if (c0 >= 0xF0) len = 4;
      else if (c0 >= 0xE0) len = 3;
      else if (c0 >= 0xC0) len = 2;
      len = std::min(len, gr.size() - i);
      const std::string_view ch = gr.substr(i, len);
      if (!lookup(ch, out)) out.append(ch);
      i += len;
    }
  }
  return out;
}

std::string Model::normalize(std::string_view raw) const {
  std::u32string cps = text::decode(raw);
  if (spec_.max_chars > 0 && cps.size() > spec_.max_chars) cps.resize(spec_.max_chars);
  std::string cur = text::encode(cps);
  for (const Op& op : spec_.ops) {
    switch (op.kind) {
      case OpKind::Bert:
        cur = bert(cur);
        break;
      case OpKind::Precompiled:
        cur = precompiled(op, cur);
        break;
      case OpKind::Replace: {
        if (op.a.empty()) break;
        std::string nxt;
        nxt.reserve(cur.size());
        for (std::size_t i = 0; i < cur.size();) {
          if (cur.compare(i, op.a.size(), op.a) == 0) {
            nxt += op.b;
            i += op.a.size();
          } else {
            nxt.push_back(cur[i++]);
          }
        }
        cur.swap(nxt);
        break;
      }
      case OpKind::Squeeze: {
        std::string nxt;
        nxt.reserve(cur.size());
        for (char c : cur) {
          if (c == ' ' && !nxt.empty() && nxt.back() == ' ') continue;
          nxt.push_back(c);
        }
        cur.swap(nxt);
        break;
      }
      case OpKind::Collapse: {
        std::u32string in32 = text::decode(cur), nxt;
        nxt.reserve(in32.size());
        bool ws = false;
        for (char32_t cp : in32) {
          if (text::space(cp)) {
            if (!ws) nxt.push_back(U' ');
            ws = true;
          } else {
            nxt.push_back(cp);
            ws = false;
          }
        }
        cur = text::encode(nxt);
        break;
      }
      case OpKind::Strip: {
        std::u32string in32 = text::decode(cur);
        std::size_t lo = 0, hi = in32.size();
        if (op.arg & 1) {
          while (lo < hi && text::space(in32[lo])) ++lo;
        }
        if (op.arg & 2) {
          while (hi > lo && text::space(in32[hi - 1])) --hi;
        }
        cur = text::encode(in32.substr(lo, hi - lo));
        break;
      }
    }
  }
  return cur;
}

std::vector<std::string> Model::pretokenize(std::string_view norm) const {
  std::vector<std::string> out;
  if (spec_.pre == Pre::Metaspace) {
    if (norm.empty()) return out;
    std::string piece;
    piece.reserve(norm.size() + spec_.meta.size());
    for (char c : norm) {
      if (c == ' ') piece += spec_.meta;
      else piece.push_back(c);
    }
    if (spec_.prepend == 1 && piece.compare(0, spec_.meta.size(), spec_.meta) != 0) piece = spec_.meta + piece;
    out.push_back(std::move(piece));
    return out;
  }
  std::string cur;
  for (char32_t cp : text::decode(norm)) {
    if (text::space(cp)) {
      if (!cur.empty()) out.push_back(std::move(cur)), cur.clear();
    } else if (text::punct(cp)) {
      if (!cur.empty()) out.push_back(std::move(cur)), cur.clear();
      std::string p;
      text::encode(cp, p);
      out.push_back(std::move(p));
    } else {
      text::encode(cp, cur);
    }
  }
  if (!cur.empty()) out.push_back(std::move(cur));
  return out;
}

std::vector<std::int32_t> Model::wordpiece(std::string_view norm) const {
  std::vector<std::int32_t> out;
  std::vector<std::size_t> cut;
  std::string key;
  for (const std::string& word : pretokenize(norm)) {
    cut.clear();
    for (std::size_t i = 0; i < word.size(); ++i) {
      if ((static_cast<unsigned char>(word[i]) & 0xC0) != 0x80) cut.push_back(i);
    }
    cut.push_back(word.size());
    const std::size_t nc = cut.size() - 1;
    if (nc > spec_.max_word) {
      out.push_back(spec_.unk);
      continue;
    }
    const std::size_t mark = out.size();
    bool bad = false;
    for (std::size_t st = 0; st < nc;) {
      std::int32_t hit = -1;
      std::size_t en = nc;
      for (; en > st; --en) {
        key.clear();
        if (st > 0) key = spec_.prefix;
        key.append(word, cut[st], cut[en] - cut[st]);
        const auto it = ids_.find(key);
        if (it != ids_.end()) {
          hit = it->second;
          break;
        }
      }
      if (hit < 0) {
        bad = true;
        break;
      }
      out.push_back(hit);
      st = en;
    }
    if (bad) {
      out.resize(mark);
      out.push_back(spec_.unk);
    }
  }
  return out;
}

void Model::prefix_hits(std::string_view s, std::size_t from,
                        std::vector<std::pair<std::size_t, std::int32_t>>& out) const {
  out.clear();
  std::uint32_t node = 0;
  for (std::size_t i = from; i < s.size(); ++i) {
    const auto it = edge_.find((std::uint64_t(node) << 8) | static_cast<unsigned char>(s[i]));
    if (it == edge_.end()) return;
    node = it->second;
    if (leaf_[node] >= 0) out.emplace_back(i + 1 - from, leaf_[node]);
  }
}

std::vector<std::int32_t> Model::unigram(std::string_view norm) const {
  std::vector<std::int32_t> out;
  for (const std::string& piece : pretokenize(norm)) {
    const std::size_t n = piece.size();
    if (n == 0) continue;
    struct Node {
      std::int32_t id = -1;
      double score = 0.0;
      std::int64_t from = -1;
    };
    std::vector<Node> best(n + 1);
    best[0].from = 0;
    std::vector<std::pair<std::size_t, std::int32_t>> hits;
    for (std::size_t st = 0; st < n;) {
      const auto c0 = static_cast<unsigned char>(piece[st]);
      std::size_t mb = c0 >= 0xF0 ? 4 : c0 >= 0xE0 ? 3 : c0 >= 0xC0 ? 2 : 1;
      mb = std::min(mb, n - st);
      const double base = best[st].score;
      bool single = false;
      prefix_hits(piece, st, hits);
      for (const auto& [len, id] : hits) {
        Node& tgt = best[st + len];
        const double cand = scores_[static_cast<std::size_t>(id)] + base;
        if (tgt.from < 0 || cand > tgt.score) {
          tgt.score = cand;
          tgt.from = static_cast<std::int64_t>(st);
          tgt.id = id;
        }
        if (len == mb) single = true;
      }
      if (!single) {
        Node& tgt = best[st + mb];
        const double cand = unk_score_ + base;
        if (tgt.from < 0 || cand > tgt.score) {
          tgt.score = cand;
          tgt.from = static_cast<std::int64_t>(st);
          tgt.id = spec_.unk;
        }
      }
      st += mb;
    }
    std::vector<std::int32_t> rev;
    bool last_unk = false;
    for (std::size_t en = n; en > 0;) {
      const Node& nd = best[en];
      if (nd.from < 0) break;
      const bool unk = nd.id == spec_.unk;
      if (!(unk && last_unk)) rev.push_back(nd.id);
      last_unk = unk;
      en = static_cast<std::size_t>(nd.from);
    }
    out.insert(out.end(), rev.rbegin(), rev.rend());
  }
  return out;
}

std::vector<std::int32_t> Model::tokenize(std::string_view raw) const {
  const std::string norm = normalize(raw);
  std::vector<std::int32_t> out = spec_.algo == Algo::Unigram ? unigram(norm) : wordpiece(norm);
  if (spec_.max_tokens > 0 && out.size() > spec_.max_tokens) out.resize(spec_.max_tokens);
  return out;
}

void Model::row(std::int32_t id, float* acc) const {
  const std::size_t dim = spec_.dim;
  const std::size_t base = static_cast<std::size_t>(id) * dim;
  if (spec_.quant == Quant::I8) {
    const float sc = scale_[static_cast<std::size_t>(id)];
    const std::int8_t* q = i8_.data() + base;
    for (std::size_t c = 0; c < dim; ++c) acc[c] += sc * static_cast<float>(q[c]);
  } else {
    const float* w = f32_.data() + base;
    for (std::size_t c = 0; c < dim; ++c) acc[c] += w[c];
  }
}

void Model::embed_into(std::string_view raw, float* out) const {
  const std::size_t dim = spec_.dim;
  std::fill(out, out + dim, 0.0f);
  std::size_t cnt = 0;
  for (std::int32_t id : tokenize(raw)) {
    if (id == spec_.unk || id < 0) continue;
    row(id, out);
    ++cnt;
  }
  if (cnt == 0) return;
  const float inv = 1.0f / static_cast<float>(cnt);
  for (std::size_t c = 0; c < dim; ++c) out[c] *= inv;
  if (spec_.flags & kNorm) {
    double nrm = 0;
    for (std::size_t c = 0; c < dim; ++c) nrm += double(out[c]) * out[c];
    if (nrm > 0) {
      const float k = static_cast<float>(1.0 / std::sqrt(nrm));
      for (std::size_t c = 0; c < dim; ++c) out[c] *= k;
    }
  }
}

std::vector<float> Model::embed(std::string_view raw) const {
  std::vector<float> out(spec_.dim);
  embed_into(raw, out.data());
  return out;
}

}
