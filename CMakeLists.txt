cmake_minimum_required(VERSION 3.20)
project(hallar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(HALLAR_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(HALLAR_SHARED "Build the shared library for FFI" ON)
option(HALLAR_SANITIZE "Build with ASan and UBSan" OFF)

set(HALLAR_SRC
  core/src/text.cpp
  core/src/model.cpp
  core/src/engine.cpp
  core/src/capi.cpp
  core/third_party/vecdex/src/index.cpp)

add_library(hallar_obj OBJECT ${HALLAR_SRC})
target_include_directories(hallar_obj PUBLIC core/include core/third_party/vecdex/include PRIVATE core/src)
target_compile_options(hallar_obj PRIVATE -fvisibility=hidden -fno-math-errno)
if(HALLAR_SANITIZE)
  target_compile_options(hallar_obj PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(hallar_obj PUBLIC -fsanitize=address,undefined)
endif()

add_library(hallar STATIC $<TARGET_OBJECTS:hallar_obj>)
target_include_directories(hallar PUBLIC core/include core/third_party/vecdex/include)
if(HALLAR_SANITIZE)
  target_link_options(hallar PUBLIC -fsanitize=address,undefined)
endif()

if(HALLAR_SHARED)
  add_library(hallar_shared SHARED $<TARGET_OBJECTS:hallar_obj>)
  set_target_properties(hallar_shared PROPERTIES OUTPUT_NAME hallar)
  if(HALLAR_SANITIZE)
    target_link_options(hallar_shared PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(HALLAR_TESTS)
  include(CTest)
  add_executable(hallar_tests core/tests/test_core.cpp)
  target_link_libraries(hallar_tests PRIVATE hallar)
  target_include_directories(hallar_tests PRIVATE core/src)
  target_compile_options(hallar_tests PRIVATE -Wall -Wextra)
  add_test(NAME core COMMAND hallar_tests)
endif()
