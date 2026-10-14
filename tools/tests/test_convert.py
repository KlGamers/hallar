import base64
import os
from pathlib import Path

import numpy as np
import pytest

from hallar_tools import convert as cv
from hallar_tools.fixture import VOCAB, build

ROOT = Path(__file__).resolve().parents[2]


def wordpiece_tok(**nrm):
    return {
        "normalizer": {"type": "BertNormalizer", "lowercase": True, "strip_accents": None, "clean_text": True,
                       "handle_chinese_chars": True, **nrm},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1}},
    }


def unigram_tok():
    return {
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Sequence", "normalizers": [
                {"type": "Precompiled", "precompiled_charsmap": base64.b64encode(b"\x00\x00\x00\x00").decode()},
                {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
            {"type": "Replace", "pattern": {"String": "!"}, "content": " ! "},
            {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
            {"type": "Strip", "strip_left": True, "strip_right": True}]},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False},
        "model": {"type": "Unigram", "unk_id": 1, "byte_fallback": False, "vocab": [["<pad>", 0.0], ["<unk>", 0.0], ["▁a", -1.0]]},
    }


def test_wordpiece_flags():
    sp = cv.spec_from(wordpiece_tok(), True, 512, 6)
    assert sp["algo"] == cv.WORDPIECE and sp["pre"] == cv.PRE_BERT
    assert sp["flags"] == cv.LOWER | cv.STRIP | cv.CJK | cv.CLEAN | cv.NORM
    assert sp["max_chars"] == 3072 and sp["unk"] == "[UNK]"
    cased = cv.spec_from(wordpiece_tok(lowercase=False), False, 0, 6)
    assert cased["flags"] == cv.CJK | cv.CLEAN and cased["max_chars"] == 0
    explicit = cv.spec_from(wordpiece_tok(lowercase=False, strip_accents=True), False, 512, 6)
    assert explicit["flags"] & cv.STRIP


def test_unigram_ops():
    sp = cv.spec_from(unigram_tok(), True, 512, 6)
    assert sp["algo"] == cv.UNIGRAM and sp["pre"] == cv.PRE_META and sp["prepend"] == 1 and sp["unk"] == "<unk>"
    kinds = [op[0] for op in sp["ops"]]
    assert kinds == [cv.OP_PRE, cv.OP_SQUEEZE, cv.OP_REPLACE, cv.OP_COLLAPSE, cv.OP_STRIP]
    assert sp["ops"][2][1:3] == (b"!", b" ! ") and sp["ops"][4][3] == 3


@pytest.mark.parametrize("mutate,msg", [
    (lambda t: t["model"].update(type="BPE"), "not supported"),
    (lambda t: t["model"].update(byte_fallback=True), "byte fallback"),
    (lambda t: t["pre_tokenizer"].update(split=True), "pre-tokenizer"),
    (lambda t: t["normalizer"]["normalizers"].append({"type": "NFKC"}), "NFKC"),
    (lambda t: t["normalizer"]["normalizers"].append({"type": "Replace", "pattern": {"Regex": "[a-z]+"}, "content": ""}), "replace pattern"),
])
def test_rejects(mutate, msg):
    tok = unigram_tok()
    mutate(tok)
    with pytest.raises(cv.Unsupported, match=msg):
        cv.spec_from(tok, True, 512, 6)


def test_wordpiece_needs_bert_pretokenizer():
    tok = wordpiece_tok()
    tok["pre_tokenizer"] = {"type": "Metaspace", "split": False}
    with pytest.raises(cv.Unsupported):
        cv.spec_from(tok, True, 512, 6)


def test_vocab_and_prune():
    words, scores = cv.vocab_from(unigram_tok())
    assert words == ["<pad>", "<unk>", "▁a"] and scores == [0.0, 0.0, -1.0]
    vocab = ["[UNK]", "hola", "##s", "##ito", "▁x", "▁casa", "perro", "q"]
    keep = cv.prune(vocab, ["##", "▁"], used={1}, keep=["[UNK]"])
    assert keep == [0, 1, 2, 4, 7]


def lib_path():
    for c in (os.environ.get("HALLAR_LIB"), ROOT / "build-rel" / "libhallar.dylib", ROOT / "build-rel" / "libhallar.so"):
        if c and Path(c).exists():
            return str(c)
    return None


@pytest.mark.skipif(lib_path() is None, reason="native library not built")
@pytest.mark.parametrize("quant", ["f32", "i8"])
def test_written_model_loads(tmp_path, quant):
    from hallar_tools.native import Model, lib

    path = tmp_path / "m.hallar"
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(VOCAB), 8)).astype(np.float32)
    sp = cv.spec_from(wordpiece_tok(), True, 512, 6)
    cv.write(path, sp, VOCAB, emb, quant, None)
    m = Model(str(path), lib(lib_path()))
    assert m.dim == 8
    assert m.tokenize("Hola mundos!") == [2, 3, 4, 23]
    v = m.embed("hola mundo")
    want = emb[[2, 3]].mean(axis=0)
    want /= np.linalg.norm(want)
    assert float(v @ want) > (0.9999 if quant == "i8" else 0.999999)


@pytest.mark.skipif(lib_path() is None, reason="native library not built")
def test_fixture(tmp_path):
    from hallar_tools.native import Engine, Model, lib

    build(str(tmp_path / "t.hallar"))
    eng = Engine(Model(str(tmp_path / "t.hallar"), lib(lib_path())))
    for i, t in enumerate(["el perro", "receta de croquetas", "boleto de tren"]):
        eng.add(str(i), t)
    assert eng.search("croquetas", 1, mode="keyword")[0][0] == "1"
    assert eng.search("tren", 1)[0][0] == "2"
