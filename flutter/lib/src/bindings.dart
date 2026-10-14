import 'dart:ffi';

final class HitC extends Struct {
  external Pointer<Char> id;
  external Pointer<Char> text;
  @Float()
  external double score;
}

final class HitsC extends Struct {
  external Pointer<HitC> hits;
  @Size()
  external int len;
}

class Bindings {
  Bindings(DynamicLibrary lib)
      : lastError = lib.lookupFunction<Pointer<Char> Function(), Pointer<Char> Function()>('hallar_last_error'),
        version = lib.lookupFunction<Pointer<Char> Function(), Pointer<Char> Function()>('hallar_version'),
        modelOpen = lib.lookupFunction<Pointer<Void> Function(Pointer<Char>), Pointer<Void> Function(Pointer<Char>)>(
            'hallar_model_open'),
        modelFree = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('hallar_model_free'),
        modelDim = lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('hallar_model_dim'),
        modelVocab = lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('hallar_model_vocab'),
        embed = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>, Pointer<Float>),
            int Function(Pointer<Void>, Pointer<Char>, Pointer<Float>)>('hallar_embed'),
        tokenize = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>, Pointer<Int32>, Size, Pointer<Size>),
            int Function(Pointer<Void>, Pointer<Char>, Pointer<Int32>, int, Pointer<Size>)>('hallar_tokenize'),
        engineNew = lib.lookupFunction<Pointer<Void> Function(Pointer<Void>, Size, Size),
            Pointer<Void> Function(Pointer<Void>, int, int)>('hallar_engine_new'),
        engineLoad = lib.lookupFunction<Pointer<Void> Function(Pointer<Void>, Pointer<Char>),
            Pointer<Void> Function(Pointer<Void>, Pointer<Char>)>('hallar_engine_load'),
        engineFree = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('hallar_engine_free'),
        engineAdd = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>, Pointer<Char>),
            int Function(Pointer<Void>, Pointer<Char>, Pointer<Char>)>('hallar_engine_add'),
        engineSearch = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>, Size, Size, Int, Pointer<HitsC>),
            int Function(Pointer<Void>, Pointer<Char>, int, int, int, Pointer<HitsC>)>('hallar_engine_search_mode'),
        engineSave = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>),
            int Function(Pointer<Void>, Pointer<Char>)>('hallar_engine_save'),
        engineSize = lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('hallar_engine_size'),
        engineHas = lib.lookupFunction<Int Function(Pointer<Void>, Pointer<Char>),
            int Function(Pointer<Void>, Pointer<Char>)>('hallar_engine_has'),
        engineSetWeight = lib.lookupFunction<Int Function(Pointer<Void>, Float),
            int Function(Pointer<Void>, double)>('hallar_engine_set_weight'),
        hitsFree = lib.lookupFunction<Void Function(Pointer<HitsC>), void Function(Pointer<HitsC>)>('hallar_hits_free');

  final Pointer<Char> Function() lastError;
  final Pointer<Char> Function() version;
  final Pointer<Void> Function(Pointer<Char>) modelOpen;
  final void Function(Pointer<Void>) modelFree;
  final int Function(Pointer<Void>) modelDim;
  final int Function(Pointer<Void>) modelVocab;
  final int Function(Pointer<Void>, Pointer<Char>, Pointer<Float>) embed;
  final int Function(Pointer<Void>, Pointer<Char>, Pointer<Int32>, int, Pointer<Size>) tokenize;
  final Pointer<Void> Function(Pointer<Void>, int, int) engineNew;
  final Pointer<Void> Function(Pointer<Void>, Pointer<Char>) engineLoad;
  final void Function(Pointer<Void>) engineFree;
  final int Function(Pointer<Void>, Pointer<Char>, Pointer<Char>) engineAdd;
  final int Function(Pointer<Void>, Pointer<Char>, int, int, int, Pointer<HitsC>) engineSearch;
  final int Function(Pointer<Void>, Pointer<Char>) engineSave;
  final int Function(Pointer<Void>) engineSize;
  final int Function(Pointer<Void>, Pointer<Char>) engineHas;
  final int Function(Pointer<Void>, double) engineSetWeight;
  final void Function(Pointer<HitsC>) hitsFree;
}
