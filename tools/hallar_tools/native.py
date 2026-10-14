from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class Hit(ctypes.Structure):
    _fields_ = [("id", ctypes.c_char_p), ("text", ctypes.c_char_p), ("score", ctypes.c_float)]


class Hits(ctypes.Structure):
    _fields_ = [("hits", ctypes.POINTER(Hit)), ("len", ctypes.c_size_t)]


def lib(path: str | None = None) -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[2]
    cand = [path, os.environ.get("HALLAR_LIB"), root / "build-rel" / "libhallar.dylib", root / "build-rel" / "libhallar.so"]
    for c in cand:
        if c and Path(c).exists():
            dll = ctypes.CDLL(str(c))
            break
    else:
        raise FileNotFoundError("libhallar not found; build it or set HALLAR_LIB")
    vp, cp, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
    sig = {
        "hallar_last_error": ([], cp),
        "hallar_model_open": ([cp], vp),
        "hallar_model_free": ([vp], None),
        "hallar_model_dim": ([vp], sz),
        "hallar_embed": ([vp, cp, ctypes.POINTER(ctypes.c_float)], ctypes.c_int),
        "hallar_tokenize": ([vp, cp, ctypes.POINTER(ctypes.c_int32), sz, ctypes.POINTER(sz)], ctypes.c_int),
        "hallar_engine_new": ([vp, sz, sz], vp),
        "hallar_engine_free": ([vp], None),
        "hallar_engine_add": ([vp, cp, cp], ctypes.c_int),
        "hallar_engine_search": ([vp, cp, sz, sz, ctypes.POINTER(Hits)], ctypes.c_int),
        "hallar_engine_search_mode": ([vp, cp, sz, sz, ctypes.c_int, ctypes.POINTER(Hits)], ctypes.c_int),
        "hallar_hits_free": ([ctypes.POINTER(Hits)], None),
        "hallar_engine_set_weight": ([vp, ctypes.c_float], ctypes.c_int),
    }
    for name, (args, res) in sig.items():
        fn = getattr(dll, name)
        fn.argtypes = args
        fn.restype = res
    return dll


class Model:
    def __init__(self, path: str, dll: ctypes.CDLL | None = None) -> None:
        self.dll = dll or lib()
        self.ptr = self.dll.hallar_model_open(str(path).encode())
        if not self.ptr:
            raise RuntimeError(self.dll.hallar_last_error().decode())
        self.dim = self.dll.hallar_model_dim(self.ptr)

    def embed(self, text: str) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        if self.dll.hallar_embed(self.ptr, text.encode(), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))):
            raise RuntimeError(self.dll.hallar_last_error().decode())
        return out

    def tokenize(self, text: str) -> list[int]:
        n = ctypes.c_size_t()
        self.dll.hallar_tokenize(self.ptr, text.encode(), None, 0, ctypes.byref(n))
        buf = (ctypes.c_int32 * max(n.value, 1))()
        self.dll.hallar_tokenize(self.ptr, text.encode(), buf, n.value, ctypes.byref(n))
        return list(buf[: n.value])

    def __del__(self) -> None:
        if getattr(self, "ptr", None):
            self.dll.hallar_model_free(self.ptr)
            self.ptr = None


class Engine:
    def __init__(self, mdl: Model) -> None:
        self.mdl = mdl
        self.ptr = mdl.dll.hallar_engine_new(mdl.ptr, 16, 200)

    def add(self, doc_id: str, text: str) -> None:
        if self.mdl.dll.hallar_engine_add(self.ptr, doc_id.encode(), text.encode()):
            raise RuntimeError(self.mdl.dll.hallar_last_error().decode())

    def set_weight(self, w: float) -> None:
        if self.mdl.dll.hallar_engine_set_weight(self.ptr, w):
            raise RuntimeError(self.mdl.dll.hallar_last_error().decode())

    def search(self, query: str, k: int = 10, ef: int = 64, mode: str = "hybrid") -> list[tuple[str, float]]:
        hs = Hits()
        code = {"hybrid": 0, "vector": 1, "keyword": 2}[mode]
        if self.mdl.dll.hallar_engine_search_mode(self.ptr, query.encode(), k, ef, code, ctypes.byref(hs)):
            raise RuntimeError(self.mdl.dll.hallar_last_error().decode())
        out = [(hs.hits[i].id.decode(), hs.hits[i].score) for i in range(hs.len)]
        self.mdl.dll.hallar_hits_free(ctypes.byref(hs))
        return out

    def __del__(self) -> None:
        if getattr(self, "ptr", None):
            self.mdl.dll.hallar_engine_free(self.ptr)
            self.ptr = None
