from __future__ import annotations

import sys

import numpy as np

from .convert import CJK, CLEAN, LOWER, NORM, OP_BERT, PRE_BERT, STRIP, WORDPIECE, write

VOCAB = ["[PAD]", "[UNK]", "hola", "mundo", "##s", "el", "la", "de", "y", "perro", "gato", "##ito", "casa",
         "cafe", "leche", "con", "receta", "croquetas", "jamon", "tren", "boleto", "viaje", "precio", "!", "?",
         ",", ".", "agua", "fria", "caliente", "noche", "dia"]


def build(path: str, dim: int = 8, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 1, (len(VOCAB), dim)).astype(np.float32)
    emb[VOCAB.index("perro")] = emb[VOCAB.index("gato")] * 0.9 + 0.1
    sp = {
        "algo": WORDPIECE, "pre": PRE_BERT, "meta": "▁", "prepend": 1,
        "ops": [(OP_BERT, b"", b"", 0)], "flags": LOWER | STRIP | CJK | CLEAN | NORM,
        "unk": "[UNK]", "prefix": "##", "max_word": 100, "max_tokens": 512, "max_chars": 3072,
    }
    write(path, sp, VOCAB, emb, "f32", None)


if __name__ == "__main__":
    build(sys.argv[1])
