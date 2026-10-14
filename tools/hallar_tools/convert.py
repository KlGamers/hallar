from __future__ import annotations

import argparse
import base64
import json
import struct
from collections.abc import Iterable
from pathlib import Path

import numpy as np

MAGIC = 0x524C4C48
VERSION = 2
LOWER, STRIP, NORM, CJK, CLEAN = 1, 2, 4, 8, 16
OP_BERT, OP_PRE, OP_REPLACE, OP_SQUEEZE, OP_COLLAPSE, OP_STRIP = range(6)
WORDPIECE, UNIGRAM = 0, 1
PRE_BERT, PRE_META = 0, 1
PREPEND = {"never": 0, "always": 1, "first": 2}


class Unsupported(ValueError):
    pass


def bert_flags(nrm: dict) -> int:
    lower = bool(nrm.get("lowercase", True))
    strip = nrm.get("strip_accents")
    strip = lower if strip is None else bool(strip)
    flags = (LOWER if lower else 0) | (STRIP if strip else 0)
    flags |= CJK if nrm.get("handle_chinese_chars", True) else 0
    flags |= CLEAN if nrm.get("clean_text", True) else 0
    return flags


def ops_from(nrm: dict | None) -> tuple[list[tuple[int, bytes, bytes, int]], int]:
    if not nrm:
        return [], 0
    t = nrm.get("type")
    if t == "Sequence":
        out, flags = [], 0
        for sub in nrm["normalizers"]:
            o, f = ops_from(sub)
            out += o
            flags |= f
        return out, flags
    if t == "BertNormalizer":
        return [(OP_BERT, b"", b"", 0)], bert_flags(nrm)
    if t == "Precompiled":
        return [(OP_PRE, base64.b64decode(nrm["precompiled_charsmap"]), b"", 0)], 0
    if t == "Replace":
        pat = nrm["pattern"]
        if "String" in pat:
            return [(OP_REPLACE, pat["String"].encode(), nrm["content"].encode(), 0)], 0
        rx = pat.get("Regex")
        if rx == " {2,}" and nrm["content"] == " ":
            return [(OP_SQUEEZE, b"", b"", 0)], 0
        if rx == "\\s+" and nrm["content"] == " ":
            return [(OP_COLLAPSE, b"", b"", 0)], 0
        raise Unsupported(f"replace pattern {pat!r} is not supported")
    if t == "Strip":
        return [(OP_STRIP, b"", b"", (1 if nrm.get("strip_left") else 0) | (2 if nrm.get("strip_right") else 0))], 0
    raise Unsupported(f"normalizer {t!r} is not supported")


def spec_from(tok: dict, normalize: bool, max_tokens: int, median: int) -> dict:
    mdl = tok["model"]
    algo = {"WordPiece": WORDPIECE, "Unigram": UNIGRAM}.get(mdl.get("type"))
    if algo is None:
        raise Unsupported(f"tokenizer model {mdl.get('type')!r} is not supported")
    if algo == UNIGRAM and mdl.get("byte_fallback"):
        raise Unsupported("unigram byte fallback is not supported")
    pre = tok.get("pre_tokenizer") or {}
    if pre.get("type") == "BertPreTokenizer":
        pre_kind, meta, prepend = PRE_BERT, "▁", 1
    elif pre.get("type") == "Metaspace" and not pre.get("split", True):
        pre_kind, meta = PRE_META, pre.get("replacement", "▁")
        prepend = PREPEND[pre.get("prepend_scheme", "always")]
    else:
        raise Unsupported(f"pre-tokenizer {pre!r} is not supported")
    ops, flags = ops_from(tok.get("normalizer"))
    if algo == WORDPIECE and pre_kind != PRE_BERT:
        raise Unsupported("WordPiece is only supported with BertPreTokenizer")
    flags |= NORM if normalize else 0
    if algo == WORDPIECE:
        unk = mdl["unk_token"]
    else:
        unk = mdl["vocab"][mdl["unk_id"]][0] if mdl.get("unk_id") is not None else None
    return {
        "algo": algo,
        "pre": pre_kind,
        "meta": meta,
        "prepend": prepend,
        "ops": ops,
        "flags": flags,
        "unk": unk,
        "prefix": mdl.get("continuing_subword_prefix", "##") or "",
        "max_word": int(mdl.get("max_input_chars_per_word", 100) or 100),
        "max_tokens": max_tokens,
        "max_chars": max_tokens * median if max_tokens else 0,
    }


def vocab_from(tok: dict) -> tuple[list[str], list[float] | None]:
    mdl = tok["model"]
    if mdl["type"] == "Unigram":
        return [t for t, _ in mdl["vocab"]], [float(s) for _, s in mdl["vocab"]]
    out = [None] * len(mdl["vocab"])
    for t, i in mdl["vocab"].items():
        out[i] = t
    return out, None


def prune(vocab: list[str], prefixes: Iterable[str], used: set[int], keep: Iterable[str]) -> list[int]:
    must = set(keep)
    pre = [p for p in prefixes if p]
    out = []
    for i, tok in enumerate(vocab):
        core = tok
        for p in pre:
            if core.startswith(p) and len(core) > len(p):
                core = core[len(p) :]
                break
        if i in used or tok in must or len(core) == 1:
            out.append(i)
    return out


def write(path: str | Path, sp: dict, vocab: list[str], emb: np.ndarray, quant: str,
          scores: list[float] | None) -> int:
    dim = emb.shape[1]
    unk = vocab.index(sp["unk"]) if sp["unk"] in vocab else -1
    def s(b: bytes) -> bytes:
        return struct.pack("<I", len(b)) + b
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IIIIiIIIII", MAGIC, VERSION, dim, len(vocab), unk, sp["flags"],
                             sp["max_tokens"], sp["max_chars"], sp["max_word"], 1 if quant == "i8" else 0))
        fh.write(s(sp["prefix"].encode()))
        fh.write(struct.pack("<II", sp["algo"], sp["pre"]))
        fh.write(s(sp["meta"].encode()))
        fh.write(struct.pack("<II", sp["prepend"], len(sp["ops"])))
        for kind, a, b, arg in sp["ops"]:
            fh.write(struct.pack("<I", kind) + s(a) + s(b) + struct.pack("<I", arg))
        for tok in vocab:
            fh.write(s(tok.encode()))
        if sp["algo"] == UNIGRAM:
            fh.write(np.asarray(scores, dtype=np.float64).tobytes())
        if quant == "i8":
            mx = np.abs(emb).max(axis=1)
            scale = np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)
            fh.write(scale.tobytes())
            fh.write(q.tobytes())
        else:
            fh.write(emb.astype(np.float32).tobytes())
    return Path(path).stat().st_size


def corpus_ids(tokenizer, files: list[str]) -> set[int]:
    used: set[int] = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        for i in range(0, len(lines), 2048):
            for enc in tokenizer.encode_batch(lines[i : i + 2048], add_special_tokens=False):
                used.update(enc.ids)
    return used


def convert(name: str, out: str, quant: str = "f32", keep_from: list[str] | None = None,
            max_tokens: int = 512) -> dict:
    from model2vec import StaticModel

    sm = StaticModel.from_pretrained(name)
    if sm.token_mapping is not None or sm.weights is not None:
        raise Unsupported("models with token mapping or per-token weights are not supported yet")
    tok = json.loads(sm.tokenizer.to_str())
    sp = spec_from(tok, bool(sm.normalize), max_tokens, int(sm.median_token_length))
    vocab, scores = vocab_from(tok)
    emb = np.asarray(sm.embedding, dtype=np.float32)
    if len(vocab) != emb.shape[0] or any(v is None for v in vocab):
        raise Unsupported("vocabulary and embedding rows do not line up")
    rows = list(range(len(vocab)))
    if keep_from:
        special = [t["content"] for t in tok.get("added_tokens", [])] + ([sp["unk"]] if sp["unk"] else [])
        rows = prune(vocab, [sp["prefix"], sp["meta"]], corpus_ids(sm.tokenizer, keep_from), special)
    size = write(out, sp, [vocab[i] for i in rows], emb[rows], quant,
                 [scores[i] for i in rows] if scores is not None else None)
    algo = "unigram" if sp["algo"] == UNIGRAM else "wordpiece"
    return {"algo": algo, "vocab": len(rows), "of": len(vocab), "dim": int(emb.shape[1]), "bytes": size,
            "ops": len(sp["ops"]), "max_tokens": max_tokens}


def main() -> None:
    par = argparse.ArgumentParser(description="Convert a model2vec static model into a .hallar file")
    par.add_argument("model", help="Hugging Face id or local path")
    par.add_argument("out")
    par.add_argument("--quant", choices=["f32", "i8"], default="f32")
    par.add_argument("--keep-from", nargs="*", help="text files; keep only tokens they use")
    par.add_argument("--max-tokens", type=int, default=512)
    args = par.parse_args()
    info = convert(args.model, args.out, args.quant, args.keep_from, args.max_tokens)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
