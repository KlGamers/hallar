from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bench_retrieval import K, beir, half, ndcg, xquad
from hallar_tools.native import Engine, Model


def score(path: str, data: dict) -> dict:
    mdl = Model(path)
    out = {}
    for name, (docs, qs, rel) in data.items():
        eng = Engine(mdl)
        for did, txt in docs.items():
            eng.add(did, txt)
        vals = [ndcg([h[0] for h in eng.search(q, k=K, ef=128, mode="vector")], rel[qid])
                for qid, q in qs.items() if half(qid) == "test"]
        out[name] = float(np.mean(vals))
    return out


def main() -> None:
    data = {"XQuAD-es": xquad("data/xquad.es.json"), "XPQA-es": beir("data/XPQARetrieval-spa-spa")}
    rows = ["| model | size MB | vocab kept | XQuAD-es nDCG@10 | XPQA-es nDCG@10 |", "|---|---|---|---|---|"]
    for spec in sys.argv[1:]:
        label, path = spec.split("=", 1)
        mdl = Model(path)
        r = score(path, data)
        rows.append(f"| {label} | {Path(path).stat().st_size / 1e6:.1f} | {mdl.dll.hallar_model_vocab(mdl.ptr) if hasattr(mdl.dll, 'hallar_model_vocab') else '-'} "
                    f"| {r['XQuAD-es']:.3f} | {r['XPQA-es']:.3f} |")
        print(rows[-1], flush=True)
    Path("bench/SIZE_ES.md").write_text("# Compression cost, vector search only, test halves\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
