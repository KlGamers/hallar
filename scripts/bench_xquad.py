from __future__ import annotations

import argparse
import json
import math
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hallar_tools.native import Engine, Model

WORD = re.compile(r"\w+", re.UNICODE)


def load(path: str) -> tuple[list[str], list[tuple[str, int]]]:
    data = json.load(open(path, encoding="utf-8"))
    ctx, qs = [], []
    for art in data["data"]:
        for par in art["paragraphs"]:
            ctx.append(par["context"])
            for qa in par["qas"]:
                qs.append((qa["question"], len(ctx) - 1))
    return ctx, qs


def score(ranks: list[int]) -> dict:
    r = np.asarray(ranks)
    return {
        "r1": float((r == 0).mean()),
        "r5": float((r < 5).mean()),
        "mrr10": float(np.where(r < 10, 1.0 / (r + 1), 0.0).mean()),
    }


def bm25(ctx: list[str], qs: list[tuple[str, int]], k1: float = 1.2, b: float = 0.75) -> dict:
    docs = [Counter(w.lower() for w in WORD.findall(c)) for c in ctx]
    lens = np.array([sum(d.values()) for d in docs], dtype=np.float64)
    avg = lens.mean()
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    ranks, ms = [], []
    for q, gold in qs:
        t0 = time.perf_counter()
        terms = [w.lower() for w in WORD.findall(q)]
        sc = np.zeros(n)
        for w in terms:
            if w not in idf:
                continue
            for i, d in enumerate(docs):
                f = d.get(w)
                if f:
                    sc[i] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
        order = np.argsort(-sc)
        ms.append((time.perf_counter() - t0) * 1000)
        ranks.append(int(np.where(order == gold)[0][0]))
    return {**score(ranks), "p50_ms": float(np.median(ms))}


def ref(name: str, ctx: list[str], qs: list[tuple[str, int]]) -> dict:
    from model2vec import StaticModel

    sm = StaticModel.from_pretrained(name)
    c = sm.encode(ctx)
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    ranks = []
    for q, gold in qs:
        v = sm.encode([q])[0]
        v = v / max(np.linalg.norm(v), 1e-12)
        order = np.argsort(-(c @ v))
        ranks.append(int(np.where(order == gold)[0][0]))
    return score(ranks)


def native(path: str, ctx: list[str], qs: list[tuple[str, int]], mode: str = "vector") -> dict:
    eng = Engine(Model(path))
    for i, c in enumerate(ctx):
        eng.add(str(i), c)
    ranks, ms = [], []
    for q, gold in qs:
        t0 = time.perf_counter()
        hits = eng.search(q, k=len(ctx), ef=max(64, len(ctx)), mode=mode)
        ms.append((time.perf_counter() - t0) * 1000)
        ids = [int(h[0]) for h in hits]
        ranks.append(ids.index(gold) if gold in ids else len(ctx))
    return {**score(ranks), "p50_ms": float(np.median(ms)), "mb": Path(path).stat().st_size / 1e6}


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--data", default="data/xquad.es.json")
    par.add_argument("--ref", default="minishlab/M2V_multilingual_output")
    par.add_argument("--models", nargs="*", default=[])
    par.add_argument("--out", default="bench/XQUAD_ES.md")
    args = par.parse_args()
    ctx, qs = load(args.data)
    rows = ["| system | size MB | R@1 | R@5 | MRR@10 | p50 ms |", "|---|---|---|---|---|---|"]
    b = bm25(ctx, qs)
    rows.append(f"| BM25 (keywords) | - | {b['r1']:.3f} | {b['r5']:.3f} | {b['mrr10']:.3f} | {b['p50_ms']:.2f} |")
    print(rows[-1], flush=True)
    r = ref(args.ref, ctx, qs)
    rows.append(f"| model2vec in Python, full | - | {r['r1']:.3f} | {r['r5']:.3f} | {r['mrr10']:.3f} | - |")
    print(rows[-1], flush=True)
    for spec in args.models:
        label, rest = spec.split("=", 1)
        path, _, mode = rest.partition(":")
        m = native(path, ctx, qs, mode or "vector")
        rows.append(f"| {label} | {m['mb']:.1f} | {m['r1']:.3f} | {m['r5']:.3f} | {m['mrr10']:.3f} | {m['p50_ms']:.2f} |")
        print(rows[-1], flush=True)
    head = [
        "# XQuAD-es passage retrieval",
        "",
        f"- {len(ctx)} Spanish paragraphs, {len(qs)} questions; the gold paragraph is the one the question was written for",
        f"- machine: {platform.machine()}, {platform.system()}, Python {platform.python_version()}",
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(head + rows) + "\n")


if __name__ == "__main__":
    main()
