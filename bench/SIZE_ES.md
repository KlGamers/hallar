# Compression cost, vector search only, test halves

| model | size MB | vocab kept | XQuAD-es nDCG@10 | XPQA-es nDCG@10 |
|---|---|---|---|---|
| completo, 32 bits | 524.5 | 500353 | 0.831 | 0.452 |
| recortado a español, 32 bits | 96.8 | 92613 | 0.831 | 0.442 |
| recortado a español, 8 bits | 26.1 | 92613 | 0.831 | 0.442 |
