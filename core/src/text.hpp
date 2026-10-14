#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace hallar::text {

const char* version();
bool space(char32_t cp);
bool control(char32_t cp);
bool punct(char32_t cp);
bool mark(char32_t cp);
bool cjk(char32_t cp);
void lower(char32_t cp, std::u32string& out);
void nfd(char32_t cp, std::u32string& out);
std::u32string decode(std::string_view s);
void encode(char32_t cp, std::string& out);
std::string encode(const std::u32string& s);
std::vector<std::size_t> graphemes(std::string_view s);

}
