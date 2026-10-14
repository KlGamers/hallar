from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hallar_tools.native import Engine, Model

RRF = 60.0
POOL = 50
K = 10


def half(qid: str) -> str:
    return "dev" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def beir(prefix: str) -> tuple[dict, dict, dict]:
    corpus = pd.read_parquet(prefix + "-corpus.parquet")
    queries = pd.read_parquet(prefix + "-queries.parquet")
    qrels = pd.read_parquet(prefix + "-qrels.parquet")
    docs = {}
    for r in corpus.itertuples():
        title = getattr(r, "title", "") or ""
        docs[str(r.id)] = (title + ". " + r.text) if title else r.text
    qs = {str(r.id): r.text for r in queries.itertuples()}
    rel = defaultdict(set)
    for r in qrels.itertuples():
        if r.score > 0:
            rel[str(getattr(r, "_1"))].add(str(getattr(r, "_2")))
    return docs, {q: t for q, t in qs.items() if q in rel}, rel


def xquad(path: str) -> tuple[dict, dict, dict]:
    data = json.load(open(path, encoding="utf-8"))
    docs, qs, rel = {}, {}, {}
    for art in data["data"]:
        for par in art["paragraphs"]:
            did = f"p{len(docs)}"
            docs[did] = par["context"]
            for qa in par["qas"]:
                qs[qa["id"]] = qa["question"]
                rel[qa["id"]] = {did}
    return docs, qs, rel


def ndcg(ranked: list[str], gold: set[str]) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:K]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), K)))
    return dcg / ideal if ideal else 0.0


def recall(ranked: list[str], gold: set[str]) -> float:
    return len(set(ranked[:K]) & gold) / len(gold)


def fuse(vec: list[str], kw: list[str], w: float) -> list[str]:
    acc: dict[str, float] = defaultdict(float)
    for i, d in enumerate(vec):
        acc[d] += w / (RRF + i + 1)
    for i, d in enumerate(kw):
        acc[d] += (1 - w) / (RRF + i + 1)
    return [d for d, _ in sorted(acc.items(), key=lambda kv: -kv[1])][:K]


def run(name: str, docs: dict, qs: dict, rel: dict, mdl: Model) -> dict:
    eng = Engine(mdl)
    t0 = time.perf_counter()
    for did, txt in docs.items():
        eng.add(did, txt)
    build = time.perf_counter() - t0
    out = {"name": name, "docs": len(docs), "queries": {}, "build_s": build, "ms": defaultdict(list)}
    for qid, q in qs.items():
        row = {}
        for mode in ("vector", "keyword", "hybrid"):
            t1 = time.perf_counter()
            hits = eng.search(q, k=POOL if mode != "hybrid" else K, ef=128, mode=mode)
            out["ms"][mode].append((time.perf_counter() - t1) * 1000)
            row[mode] = [h[0] for h in hits]
        out["queries"][qid] = row
    out["eng"] = eng
    return out


def table(sets: list[dict], rel_all: dict, w: float, part: str) -> list[str]:
    rows = [
        f"| dataset | docs | queries | BM25 nDCG@10 | vector nDCG@10 | hybrid nDCG@10 | BM25 R@10 | vector R@10 | hybrid R@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sets:
        rel = rel_all[s["name"]]
        qids = [q for q in s["queries"] if half(q) == part]
        m = {mode: ([], []) for mode in ("keyword", "vector", "hybrid")}
        for q in qids:
            r = s["queries"][q]
            ranked = {"keyword": r["keyword"], "vector": r["vector"], "hybrid": r["hybrid"]}
            for mode, lst in ranked.items():
                m[mode][0].append(ndcg(lst, rel[q]))
                m[mode][1].append(recall(lst, rel[q]))
        f = {k: (np.mean(v[0]), np.mean(v[1])) for k, v in m.items()}
        rows.append(
            f"| {s['name']} | {s['docs']:,} | {len(qids)} | {f['keyword'][0]:.3f} | {f['vector'][0]:.3f} | "
            f"{f['hybrid'][0]:.3f} | {f['keyword'][1]:.3f} | {f['vector'][1]:.3f} | {f['hybrid'][1]:.3f} |"
        )
    return rows


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--model", default="models/m2v-es-i8.hallar")
    par.add_argument("--out", default="bench/RETRIEVAL_ES.md")
    args = par.parse_args()
    mdl = Model(args.model)
    data = {
        "XQuAD-es": xquad("data/xquad.es.json"),
        "MIRACL-es": beir("data/MIRACLRetrievalHardNegatives-es"),
        "XPQA-es": beir("data/XPQARetrieval-spa-spa"),
    }
    sets = []
    for name, (docs, qs, rel) in data.items():
        sets.append(run(name, docs, qs, rel, mdl))
        print(f"{name}: {len(docs)} docs, {len(qs)} queries, built in {sets[-1]['build_s']:.1f}s", flush=True)
    rel_all = {name: d[2] for name, d in data.items()}

    grid = [round(x, 2) for x in np.linspace(0, 1, 21)]
    best = None
    for w in grid:
        sc = []
        for s in sets:
            for q, r in s["queries"].items():
                if half(q) == "dev":
                    sc.append(ndcg(fuse(r["vector"], r["keyword"], w), rel_all[s["name"]][q]))
        val = float(np.mean(sc))
        if best is None or val > best[1]:
            best = (w, val)
    w = best[0]
    print(f"vector weight tuned on dev halves: {w} (dev nDCG@10 {best[1]:.4f})", flush=True)
    for s in sets:
        s["eng"].set_weight(w)
        for q in s["queries"]:
            s["queries"][q]["hybrid"] = [h[0] for h in s["eng"].search(data[s["name"]][1][q], k=K, ef=128, mode="hybrid")]

    lat = {mode: float(np.median([x for s in sets for x in s["ms"][mode]])) for mode in ("keyword", "vector", "hybrid")}
    head = [
        "# Spanish retrieval",
        "",
        f"- model: `{Path(args.model).name}` ({Path(args.model).stat().st_size / 1e6:.1f} MB)",
        f"- queries are split in two fixed halves by a hash of their id; the hybrid vector weight ({w}) was chosen on the dev halves",
        "- numbers below are on the test halves only",
        f"- median latency per query: BM25 {lat['keyword']:.2f} ms, vector {lat['vector']:.2f} ms, hybrid {lat['hybrid']:.2f} ms",
        f"- index build: " + ", ".join(f"{s['name']} {s['build_s']:.1f} s" for s in sets),
        f"- machine: {platform.machine()}, {platform.system()}",
        "",
    ]
    rows = table(sets, rel_all, w, "test")
    print("\n".join(rows))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(head + rows) + "\n")


if __name__ == "__main__":
    main()
