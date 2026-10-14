import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'src/bindings.dart';

DynamicLibrary _open() {
  final env = Platform.environment['HALLAR_LIB'];
  if (env != null && env.isNotEmpty) return DynamicLibrary.open(env);
  if (Platform.isAndroid || Platform.isLinux) return DynamicLibrary.open('libhallar.so');
  if (Platform.isIOS || Platform.isMacOS) return DynamicLibrary.process();
  throw UnsupportedError('hallar does not support ${Platform.operatingSystem}');
}

final Bindings _b = Bindings(_open());

class HallarException implements Exception {
  HallarException(this.message);
  final String message;
  @override
  String toString() => 'HallarException: $message';
}

enum SearchMode { hybrid, vector, keyword }

class SearchResult {
  const SearchResult(this.id, this.text, this.score);
  final String id;
  final String text;
  final double score;

  @override
  bool operator ==(Object other) =>
      other is SearchResult && other.id == id && other.text == text && other.score == score;

  @override
  int get hashCode => Object.hash(id, text, score);

  @override
  String toString() => 'SearchResult($id, $score)';
}

HallarException _err() => HallarException(_b.lastError().cast<Utf8>().toDartString());

T _withStr<T>(String s, T Function(Pointer<Char>) fn) {
  final p = s.toNativeUtf8();
  try {
    return fn(p.cast());
  } finally {
    calloc.free(p);
  }
}

class HallarModel implements Finalizable {
  HallarModel._(this._ptr);

  factory HallarModel.open(String path) {
    final p = _withStr(path, _b.modelOpen);
    if (p == nullptr) throw _err();
    return HallarModel._(p);
  }

  Pointer<Void> _ptr;

  Pointer<Void> get _live {
    if (_ptr == nullptr) throw StateError('model is closed');
    return _ptr;
  }

  int get dimension => _b.modelDim(_live);
  int get vocabularySize => _b.modelVocab(_live);

  Float32List embed(String text) {
    final dim = dimension;
    final out = calloc<Float>(dim);
    try {
      final rc = _withStr(text, (t) => _b.embed(_live, t, out));
      if (rc != 0) throw _err();
      return Float32List.fromList(out.asTypedList(dim));
    } finally {
      calloc.free(out);
    }
  }

  List<int> tokenize(String text) {
    final n = calloc<Size>();
    try {
      return _withStr(text, (t) {
        if (_b.tokenize(_live, t, nullptr, 0, n) != 0) throw _err();
        final cnt = n.value;
        final buf = calloc<Int32>(cnt == 0 ? 1 : cnt);
        try {
          if (_b.tokenize(_live, t, buf, cnt, n) != 0) throw _err();
          return List<int>.of(buf.asTypedList(cnt));
        } finally {
          calloc.free(buf);
        }
      });
    } finally {
      calloc.free(n);
    }
  }

  void close() {
    if (_ptr != nullptr) _b.modelFree(_ptr);
    _ptr = nullptr;
  }
}

class HallarIndex implements Finalizable {
  HallarIndex._(this.model, this._ptr);

  factory HallarIndex(HallarModel model, {int connections = 16, int buildEffort = 200}) {
    final p = _b.engineNew(model._live, connections, buildEffort);
    if (p == nullptr) throw _err();
    return HallarIndex._(model, p);
  }

  factory HallarIndex.load(HallarModel model, String dir) {
    final p = _withStr(dir, (d) => _b.engineLoad(model._live, d));
    if (p == nullptr) throw _err();
    return HallarIndex._(model, p);
  }

  final HallarModel model;
  Pointer<Void> _ptr;

  Pointer<Void> get _live {
    if (_ptr == nullptr) throw StateError('index is closed');
    return _ptr;
  }

  int get length => _b.engineSize(_live);

  bool contains(String id) => _withStr(id, (p) => _b.engineHas(_live, p) == 1);

  void add(String id, String text) {
    final rc = _withStr(id, (i) => _withStr(text, (t) => _b.engineAdd(_live, i, t)));
    if (rc != 0) throw _err();
  }

  void addAll(Map<String, String> docs) => docs.forEach(add);

  void setVectorWeight(double w) {
    if (_b.engineSetWeight(_live, w) != 0) throw _err();
  }

  List<SearchResult> search(String query, {int limit = 10, SearchMode mode = SearchMode.hybrid, int effort = 64}) {
    final hits = calloc<HitsC>();
    try {
      final rc = _withStr(query, (q) => _b.engineSearch(_live, q, limit, effort, mode.index, hits));
      if (rc != 0) throw _err();
      final out = <SearchResult>[];
      for (var i = 0; i < hits.ref.len; i++) {
        final h = hits.ref.hits[i];
        out.add(SearchResult(h.id.cast<Utf8>().toDartString(), h.text.cast<Utf8>().toDartString(), h.score));
      }
      return out;
    } finally {
      _b.hitsFree(hits);
      calloc.free(hits);
    }
  }

  void save(String dir) {
    if (_withStr(dir, (d) => _b.engineSave(_live, d)) != 0) throw _err();
  }

  void close() {
    if (_ptr != nullptr) _b.engineFree(_ptr);
    _ptr = nullptr;
  }
}

String get hallarVersion => _b.version().cast<Utf8>().toDartString();
