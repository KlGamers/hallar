# Spanish retrieval

- model: `potion-es-i8.hallar` (26.1 MB)
- queries are split in two fixed halves by a hash of their id; the hybrid vector weight (0.15) was chosen on the dev halves
- numbers below are on the test halves only
- median latency per query: BM25 0.08 ms, vector 0.14 ms, hybrid 0.14 ms
- index build: XQuAD-es 0.2 s, MIRACL-es 146.6 s, XPQA-es 0.3 s
- machine: arm64, Darwin

| dataset | docs | queries | BM25 nDCG@10 | vector nDCG@10 | hybrid nDCG@10 | BM25 R@10 | vector R@10 | hybrid R@10 |
|---|---|---|---|---|---|---|---|---|
| XQuAD-es | 240 | 586 | 0.950 | 0.831 | 0.947 | 0.983 | 0.939 | 0.988 |
| MIRACL-es | 146,750 | 321 | 0.252 | 0.136 | 0.258 | 0.291 | 0.184 | 0.310 |
| XPQA-es | 1,941 | 402 | 0.463 | 0.442 | 0.484 | 0.512 | 0.503 | 0.533 |
