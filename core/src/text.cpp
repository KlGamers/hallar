#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "text.hpp"

namespace hallar::text {

namespace {

struct Range {
  char32_t lo;
  char32_t hi;
};

struct Map3 {
  char32_t cp;
  char32_t to[3];
};

struct Map4 {
  char32_t cp;
  char32_t to[4];
};

struct Prop {
  char32_t lo;
  char32_t hi;
  int gb;
};

#include "unicode_tables.inc"

template <std::size_t N>
bool in(const Range (&tbl)[N], char32_t cp) {
  const Range* end = tbl + N;
  const Range* it = std::upper_bound(tbl, end, cp, [](char32_t v, const Range& r) { return v < r.lo; });
  return it != tbl && cp <= (it - 1)->hi;
}

template <typename T, std::size_t N>
const T* find(const T (&tbl)[N], char32_t cp) {
  const T* end = tbl + N;
  const T* it = std::lower_bound(tbl, end, cp, [](const T& m, char32_t v) { return m.cp < v; });
  return it != end && it->cp == cp ? it : nullptr;
}

}

const char* version() { return kUnicodeVersion; }

namespace {

enum Gb { Other, CR, LF, Control, Extend, ZWJ, RI, Prepend, SpacingMark, L, V, T, LV, LVT };

int gb(char32_t cp) {
  const Prop* end = kGrapheme + sizeof(kGrapheme) / sizeof(kGrapheme[0]);
  const Prop* it = std::upper_bound(kGrapheme, end, cp, [](char32_t v, const Prop& r) { return v < r.lo; });
  return it != kGrapheme && cp <= (it - 1)->hi ? (it - 1)->gb : Other;
}

bool pict(char32_t cp) { return in(kPict, cp); }

bool joins(int a, int b, bool pict_zwj, int ri_run) {
  if (a == CR && b == LF) return true;
  if (a == Control || a == CR || a == LF || b == Control || b == CR || b == LF) return false;
  if (a == L && (b == L || b == V || b == LV || b == LVT)) return true;
  if ((a == LV || a == V) && (b == V || b == T)) return true;
  if ((a == LVT || a == T) && b == T) return true;
  if (b == Extend || b == ZWJ || b == SpacingMark || a == Prepend) return true;
  if (a == ZWJ && pict_zwj) return true;
  if (a == RI && b == RI) return ri_run % 2 == 1;
  return false;
}

}

std::vector<std::size_t> graphemes(std::string_view s) {
  std::vector<std::size_t> cut;
  const std::u32string cps = decode(s);
  std::size_t off = 0;
  int prev = -1;
  bool in_pict = false;
  bool pict_zwj = false;
  int ri_run = 0;
  for (char32_t cp : cps) {
    const int cur = gb(cp);
    const bool is_pict = pict(cp);
    if (prev < 0 || !joins(prev, cur, pict_zwj && is_pict, ri_run)) {
      cut.push_back(off);
      ri_run = 0;
      in_pict = false;
    }
    if (is_pict) {
      in_pict = true;
      pict_zwj = false;
    } else if (cur == ZWJ) {
      pict_zwj = in_pict;
    } else if (cur != Extend) {
      in_pict = false;
      pict_zwj = false;
    }
    ri_run = cur == RI ? ri_run + 1 : 0;
    std::string tmp;
    encode(cp, tmp);
    off += tmp.size();
    prev = cur;
  }
  cut.push_back(off);
  return cut;
}

bool space(char32_t cp) { return in(kSpace, cp); }
bool control(char32_t cp) { return in(kControl, cp); }
bool punct(char32_t cp) { return in(kPunct, cp); }
bool mark(char32_t cp) { return in(kMark, cp); }

bool cjk(char32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

void lower(char32_t cp, std::u32string& out) {
  if (const Map3* m = find(kLower, cp)) {
    for (char32_t c : m->to) {
      if (c) out.push_back(c);
    }
    return;
  }
  out.push_back(cp);
}

void nfd(char32_t cp, std::u32string& out) {
  if (const Map4* m = find(kNfd, cp)) {
    for (char32_t c : m->to) {
      if (c) out.push_back(c);
    }
    return;
  }
  out.push_back(cp);
}

std::u32string decode(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const std::size_t n = s.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = p[i];
    char32_t cp = 0xFFFD;
    std::size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6 && i + 1 < n && (p[i + 1] & 0xC0) == 0x80) {
      cp = (char32_t(c & 0x1F) << 6) | (p[i + 1] & 0x3F);
      len = 2;
      if (cp < 0x80) cp = 0xFFFD;
    } else if ((c >> 4) == 0xE && i + 2 < n && (p[i + 1] & 0xC0) == 0x80 && (p[i + 2] & 0xC0) == 0x80) {
      cp = (char32_t(c & 0x0F) << 12) | (char32_t(p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
      len = 3;
      if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
    } else if ((c >> 3) == 0x1E && i + 3 < n && (p[i + 1] & 0xC0) == 0x80 && (p[i + 2] & 0xC0) == 0x80 &&
               (p[i + 3] & 0xC0) == 0x80) {
      cp = (char32_t(c & 0x07) << 18) | (char32_t(p[i + 1] & 0x3F) << 12) | (char32_t(p[i + 2] & 0x3F) << 6) |
           (p[i + 3] & 0x3F);
      len = 4;
      if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void encode(char32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string encode(const std::u32string& s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t cp : s) encode(cp, out);
  return out;
}

}
