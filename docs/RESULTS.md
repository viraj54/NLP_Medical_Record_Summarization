# Recorded results

Values below are transcribed directly from the supplied notebook's saved aggregate output and rounded to four decimals. They were not recomputed from predictions: per-example prediction exports and the fine-tuned checkpoint were not supplied.

| Metric | Extractive routine | Pretrained BART | Fine-tuned BART |
|---|---:|---:|---:|
| ROUGE-1 | 0.2918 | 0.1926 | 0.4922 |
| ROUGE-2 | 0.1751 | 0.1169 | 0.3381 |
| ROUGE-L | 0.1936 | 0.1384 | 0.347 |
| BERTScore-F1 | 0.7291 | 0.7128 | 0.9187 |
| METEOR | 0.0684 | 0.0395 | 0.2546 |
| Keyword preservation (regex) | 0.2459 | 0.1814 | 0.5818 |
| Source-token overlap | 1.0 | 0.9995 | 0.5802 |
| Lexicon concept F1 | 0.5143 | 0.4268 | 0.6531 |

Fine-tuned BART has the largest values for the reported reference-overlap and semantic metrics, regex keyword preservation and lexicon F1. Its source-token overlap is lower than both baselines. None of these aggregate values establishes clinical reliability.

The evaluation uses 50 rows from a 750-row test split. This repository reports the sample size and does not convert these scores into an “accuracy percentage.” The unequal baseline/fine-tuned input and decoding limits are documented in METHODS.md.

The original paper's claims about low hallucination, validated clinical quality and universal improvement are not carried forward. It is replaced here by code-based methods, metric definitions and this results record. No source notes, generated clinical examples, original notebook output images or fine-tuned weights are distributed.
