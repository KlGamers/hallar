cmake_minimum_required(VERSION 3.18)
project(hallar_flutter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../core)

add_library(hallar SHARED
  ${CORE}/src/text.cpp
  ${CORE}/src/model.cpp
  ${CORE}/src/engine.cpp
  ${CORE}/src/capi.cpp
  ${CORE}/third_party/vecdex/src/index.cpp)

target_include_directories(hallar PRIVATE ${CORE}/include ${CORE}/src ${CORE}/third_party/vecdex/include)
target_compile_options(hallar PRIVATE -fvisibility=hidden -fno-math-errno)
if(ANDROID)
  target_link_options(hallar PRIVATE -Wl,-z,max-page-size=16384)
endif()
