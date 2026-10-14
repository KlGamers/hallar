cmake_minimum_required(VERSION 3.18)
project(hallar_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core)

add_library(hallar_jni SHARED
  jni.cpp
  ${CORE}/src/text.cpp
  ${CORE}/src/model.cpp
  ${CORE}/src/engine.cpp
  ${CORE}/third_party/vecdex/src/index.cpp)

target_include_directories(hallar_jni PRIVATE ${CORE}/include ${CORE}/src ${CORE}/third_party/vecdex/include)
target_compile_options(hallar_jni PRIVATE -fvisibility=hidden -fno-math-errno)

if(HALLAR_HOST)
  find_package(JNI REQUIRED COMPONENTS JVM)
  target_include_directories(hallar_jni PRIVATE ${JNI_INCLUDE_DIRS})
else()
  target_link_options(hallar_jni PRIVATE -Wl,-z,max-page-size=16384)
endif()
