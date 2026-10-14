from __future__ import annotations

import argparse
import json

import numpy as np

from .native import Model

STRESS = [
    "",
    "   ",
    "Café con leche, ¿por favor? ¡Gracias!",
    "ÁRBOL Ñandú pingüino",
    "数学 and 日本語のテキスト",
    "emoji 🚀🔥 test",
    "tabs\tand\nnewlines\r\nmixed",
    "x" * 150,
    "word " * 700,
    "U.S.A. e-mail: someone@example.com $100.00 #hash @user",
    "İstanbul ΣΑΣ straße ǅemal",
    "zero​width nbsp sep",
    "control\x01\x07chars",
    "naïve résumé coöperate",
    "﷽ العربية עברית",
]


def compare(name: str, path: str, texts: list[str]) -> dict:
    from model2vec import StaticModel

    ref = StaticModel.from_pretrained(name)
    mine = Model(path)
    tok_ok = 0
    cos = []
    bad = []
    for t in texts:
        want = [i for i in ref.tokenize([t])[0]]
        got = [i for i in mine.tokenize(t) if i != ref.unk_token_id]
        if want == got:
            tok_ok += 1
        elif len(bad) < 5:
            bad.append(t[:80])
        a = ref.encode([t])[0].astype(np.float64)
        b = mine.embed(t).astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos.append(1.0 if na == 0 and nb == 0 else float(a @ b / (na * nb)) if na and nb else 0.0)
    cos = np.asarray(cos)
    return {
        "texts": len(texts),
        "token_match": tok_ok / len(texts),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "examples_mismatch": bad,
    }


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("model")
    par.add_argument("path")
    par.add_argument("--texts", nargs="*", default=[])
    par.add_argument("--limit", type=int, default=20000)
    args = par.parse_args()
    texts = list(STRESS)
    for f in args.texts:
        with open(f, encoding="utf-8") as fh:
            texts += [ln.rstrip("\n") for ln in fh][: args.limit]
    print(json.dumps(compare(args.model, args.path, texts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
