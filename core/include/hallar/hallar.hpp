#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vecdex {
class Index;
}

namespace hallar {

enum Flag : std::uint32_t {
  kLower = 1u << 0,
  kStrip = 1u << 1,
  kNorm = 1u << 2,
  kCjk = 1u << 3,
  kClean = 1u << 4,
};

enum class Quant : std::uint32_t { F32 = 0, I8 = 1 };
enum class Algo : std::uint32_t { WordPiece = 0, Unigram = 1 };
enum class Pre : std::uint32_t { Bert = 0, Metaspace = 1 };
enum class OpKind : std::uint32_t { Bert = 0, Precompiled = 1, Replace = 2, Squeeze = 3, Collapse = 4, Strip = 5 };

struct Op {
  OpKind kind = OpKind::Bert;
  std::string a;
  std::string b;
  std::uint32_t arg = 0;
};

struct Spec {
  std::uint32_t dim = 0;
  std::int32_t unk = -1;
  std::uint32_t flags = kLower | kStrip | kNorm | kCjk | kClean;
  std::uint32_t max_tokens = 512;
  std::uint32_t max_chars = 0;
  std::uint32_t max_word = 100;
  Quant quant = Quant::F32;
  std::string prefix = "##";
  Algo algo = Algo::WordPiece;
  Pre pre = Pre::Bert;
  std::string meta = "\xE2\x96\x81";
  std::uint32_t prepend = 1;
  std::vector<Op> ops = {Op{}};
};

class Model {
 public:
  static std::shared_ptr<const Model> load(const std::string& path);
  static std::shared_ptr<const Model> build(Spec spec, std::vector<std::string> vocab,
                                            std::vector<float> weights, std::vector<double> scores = {});
  void save(const std::string& path) const;

  std::string normalize(std::string_view text) const;
  std::vector<std::string> pretokenize(std::string_view text) const;
  std::vector<std::int32_t> tokenize(std::string_view text) const;
  std::vector<float> embed(std::string_view text) const;
  void embed_into(std::string_view text, float* out) const;

  std::size_t dim() const { return spec_.dim; }
  std::size_t vocab() const { return vocab_.size(); }
  const Spec& spec() const { return spec_; }
  const std::string& token(std::int32_t id) const { return vocab_.at(static_cast<std::size_t>(id)); }
  std::uint64_t hash() const { return hash_; }

 private:
  Model() = default;
  void index_vocab();
  void row(std::int32_t id, float* acc) const;
  std::string bert(std::string_view s) const;
  std::string precompiled(const Op& op, std::string_view s) const;
  std::vector<std::int32_t> wordpiece(std::string_view norm) const;
  std::vector<std::int32_t> unigram(std::string_view norm) const;
  void prefix_hits(std::string_view s, std::size_t from, std::vector<std::pair<std::size_t, std::int32_t>>& out) const;

  Spec spec_;
  std::vector<std::string> vocab_;
  std::unordered_map<std::string, std::int32_t> ids_;
  std::vector<float> f32_;
  std::vector<std::int8_t> i8_;
  std::vector<float> scale_;
  std::vector<double> scores_;
  double unk_score_ = 0.0;
  std::unordered_map<std::uint64_t, std::uint32_t> edge_;
  std::vector<std::int32_t> leaf_;
  std::uint64_t hash_ = 0;
};

struct Hit {
  std::string id;
  float score;
  std::string text;
};

enum class Mode : std::uint32_t { Hybrid = 0, Vector = 1, Keyword = 2 };

constexpr float kDefaultWeight = 0.15f;

std::vector<std::string> terms(std::string_view text);

class Engine {
 public:
  explicit Engine(std::shared_ptr<const Model> mdl, std::size_t m = 16, std::size_t ef_build = 200);
  ~Engine();
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  void add(const std::string& id, const std::string& text);
  std::vector<Hit> search(std::string_view query, std::size_t k = 10, std::size_t ef = 64,
                          Mode mode = Mode::Hybrid) const;
  bool has(const std::string& id) const;
  void set_weight(float vec_weight);
  float weight() const;
  std::size_t size() const;

  void save(const std::string& dir) const;
  static Engine load(std::shared_ptr<const Model> mdl, const std::string& dir);

 private:
  std::shared_ptr<const Model> mdl_;
  std::unique_ptr<vecdex::Index> idx_;
  std::vector<std::string> ids_;
  std::vector<std::string> texts_;
  std::unordered_map<std::string, std::uint64_t> slot_;
  std::unordered_map<std::string, std::vector<std::pair<std::uint32_t, std::uint32_t>>> post_;
  std::vector<std::uint32_t> lens_;
  std::uint64_t total_ = 0;
  float wvec_ = kDefaultWeight;
  std::unique_ptr<std::shared_mutex> mu_;

  void index_terms(std::uint32_t doc, const std::string& text);
  std::vector<std::pair<std::uint32_t, float>> vector_rank(std::string_view q, std::size_t n, std::size_t ef) const;
  std::vector<std::pair<std::uint32_t, float>> keyword_rank(std::string_view q, std::size_t n) const;
};

const char* unicode_version();

}
