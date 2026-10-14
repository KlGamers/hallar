#include <jni.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hallar/hallar.hpp"

namespace {

using hallar::Engine;
using hallar::Model;

struct ModelBox {
  std::shared_ptr<const Model> ptr;
};

std::string utf8(JNIEnv* env, jstring s) {
  if (!s) throw std::invalid_argument("hallar: null string");
  const jsize n = env->GetStringLength(s);
  const jchar* u = env->GetStringChars(s, nullptr);
  std::string out;
  out.reserve(static_cast<std::size_t>(n));
  for (jsize i = 0; i < n; ++i) {
    std::uint32_t cp = u[i];
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n && u[i + 1] >= 0xDC00 && u[i + 1] <= 0xDFFF) {
      cp = 0x10000 + ((cp - 0xD800) << 10) + (u[i + 1] - 0xDC00);
      ++i;
    } else if (cp >= 0xD800 && cp <= 0xDFFF) {
      cp = 0xFFFD;
    }
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
  env->ReleaseStringChars(s, u);
  return out;
}

jstring jstr(JNIEnv* env, const std::string& s) {
  std::vector<jchar> u;
  u.reserve(s.size());
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  for (std::size_t i = 0; i < s.size();) {
    std::uint32_t cp = p[i];
    std::size_t len = 1;
    if (cp >= 0xF0 && i + 3 < s.size()) {
      cp = ((cp & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
      len = 4;
    } else if (cp >= 0xE0 && i + 2 < s.size()) {
      cp = ((cp & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
      len = 3;
    } else if (cp >= 0xC0 && i + 1 < s.size()) {
      cp = ((cp & 0x1F) << 6) | (p[i + 1] & 0x3F);
      len = 2;
    } else if (cp >= 0x80) {
      cp = 0xFFFD;
    }
    if (cp >= 0x10000) {
      cp -= 0x10000;
      u.push_back(static_cast<jchar>(0xD800 + (cp >> 10)));
      u.push_back(static_cast<jchar>(0xDC00 + (cp & 0x3FF)));
    } else {
      u.push_back(static_cast<jchar>(cp));
    }
    i += len;
  }
  return env->NewString(u.data(), static_cast<jsize>(u.size()));
}

void raise(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("io/github/klgamers/hallar/HallarException");
  if (cls) env->ThrowNew(cls, msg);
}

template <typename T, typename F>
T guard(JNIEnv* env, T fallback, F&& fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    raise(env, e.what());
  } catch (...) {
    raise(env, "hallar: unknown error");
  }
  return fallback;
}

ModelBox* box(jlong p) { return reinterpret_cast<ModelBox*>(p); }
Engine* eng(jlong p) { return reinterpret_cast<Engine*>(p); }

}

#define JNI_FN(name) JNIEXPORT JNICALL Java_io_github_klgamers_hallar_Native_##name

extern "C" {

jlong JNI_FN(modelOpen)(JNIEnv* env, jclass, jstring path) {
  return guard<jlong>(env, 0, [&] {
    return reinterpret_cast<jlong>(new ModelBox{Model::load(utf8(env, path))});
  });
}

void JNI_FN(modelFree)(JNIEnv*, jclass, jlong p) { delete box(p); }

jint JNI_FN(modelDim)(JNIEnv*, jclass, jlong p) { return static_cast<jint>(box(p)->ptr->dim()); }

jint JNI_FN(modelVocab)(JNIEnv*, jclass, jlong p) { return static_cast<jint>(box(p)->ptr->vocab()); }

jfloatArray JNI_FN(embed)(JNIEnv* env, jclass, jlong p, jstring text) {
  return guard<jfloatArray>(env, nullptr, [&] {
    const auto v = box(p)->ptr->embed(utf8(env, text));
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(v.size()));
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(v.size()), v.data());
    return out;
  });
}

jintArray JNI_FN(tokenize)(JNIEnv* env, jclass, jlong p, jstring text) {
  return guard<jintArray>(env, nullptr, [&] {
    const auto ids = box(p)->ptr->tokenize(utf8(env, text));
    jintArray out = env->NewIntArray(static_cast<jsize>(ids.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(ids.size()), reinterpret_cast<const jint*>(ids.data()));
    return out;
  });
}

jlong JNI_FN(engineNew)(JNIEnv* env, jclass, jlong p, jint m, jint ef) {
  return guard<jlong>(env, 0, [&] {
    return reinterpret_cast<jlong>(new Engine(box(p)->ptr, static_cast<std::size_t>(m), static_cast<std::size_t>(ef)));
  });
}

jlong JNI_FN(engineLoad)(JNIEnv* env, jclass, jlong p, jstring dir) {
  return guard<jlong>(env, 0, [&] {
    return reinterpret_cast<jlong>(new Engine(Engine::load(box(p)->ptr, utf8(env, dir))));
  });
}

void JNI_FN(engineFree)(JNIEnv*, jclass, jlong p) { delete eng(p); }

void JNI_FN(engineAdd)(JNIEnv* env, jclass, jlong p, jstring id, jstring text) {
  guard<int>(env, 0, [&] {
    eng(p)->add(utf8(env, id), utf8(env, text));
    return 0;
  });
}

jobjectArray JNI_FN(engineSearch)(JNIEnv* env, jclass, jlong p, jstring query, jint k, jint ef, jint mode) {
  return guard<jobjectArray>(env, nullptr, [&] {
    if (mode < 0 || mode > 2) throw std::invalid_argument("hallar: unknown search mode");
    const auto hits = eng(p)->search(utf8(env, query), static_cast<std::size_t>(k), static_cast<std::size_t>(ef),
                                     static_cast<hallar::Mode>(mode));
    jclass cls = env->FindClass("io/github/klgamers/hallar/SearchResult");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;F)V");
    jobjectArray out = env->NewObjectArray(static_cast<jsize>(hits.size()), cls, nullptr);
    for (std::size_t i = 0; i < hits.size(); ++i) {
      jstring jid = jstr(env, hits[i].id);
      jstring jtx = jstr(env, hits[i].text);
      jobject obj = env->NewObject(cls, ctor, jid, jtx, hits[i].score);
      env->SetObjectArrayElement(out, static_cast<jsize>(i), obj);
      env->DeleteLocalRef(obj);
      env->DeleteLocalRef(jid);
      env->DeleteLocalRef(jtx);
    }
    return out;
  });
}

void JNI_FN(engineSave)(JNIEnv* env, jclass, jlong p, jstring dir) {
  guard<int>(env, 0, [&] {
    eng(p)->save(utf8(env, dir));
    return 0;
  });
}

jint JNI_FN(engineSize)(JNIEnv*, jclass, jlong p) { return static_cast<jint>(eng(p)->size()); }

jboolean JNI_FN(engineHas)(JNIEnv* env, jclass, jlong p, jstring id) {
  return guard<jboolean>(env, JNI_FALSE, [&] { return eng(p)->has(utf8(env, id)) ? JNI_TRUE : JNI_FALSE; });
}

void JNI_FN(engineSetWeight)(JNIEnv* env, jclass, jlong p, jfloat w) {
  guard<int>(env, 0, [&] {
    eng(p)->set_weight(w);
    return 0;
  });
}

}
