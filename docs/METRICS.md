# Metric definitions and interpretation

| Public label | Original notebook label | What the implementation measures |
|---|---|---|
| ROUGE-1/2/L | Same | Mean reference/prediction F-measures from `rouge-score`, with stemming |
| BERTScore-F1 | Same | Mean contextual token-matching F1 using `distilbert-base-uncased` |
| METEOR | Same | NLTK METEOR on lowercased word-tokenized prediction/reference text |
| Keyword preservation (regex) | NER Preservation | Share of regex-extracted reference keywords appearing as substrings in the prediction; the recorded run used the fallback |
| Source-token overlap | Factual Consistency | Fraction of prediction content-token occurrences also appearing somewhere in the source note |
| Lexicon concept F1 | MEDCON Score | Set F1 overlap of substring-matched terms from the notebook's custom clinical lexicon |

The lexicon calculation does not link entities to UMLS concept IDs and does not execute the official MEDCON method. Its recorded values are presented as a custom metric, not MEDCON results.

Source-token overlap removes common words including negations such as `no` and `not`. It ignores who/what a fact refers to, dose relationships and temporal order. For example, “No fever reported” versus “Fever reported” can receive full overlap even though the meaning differs. An extractive summary can score 1.0 simply because it copies source words. This metric must not be interpreted as a hallucination rate or clinical factual accuracy.

Regex keywords are not equivalent to biomedical entity recognition. Substring matching in the lexicon can also produce false positives (for example, `pain` within `painting`). The optional SciSpaCy path uses a separate extractor and is not directly comparable with the recorded regex-fallback results.

The recorded run's NER/ROUGE/METEOR functions skip some empty/error cases, and some metric functions return zero on failure. The aggregate output does not provide per-example validity counts. This limits interpretation of the effective scoring denominator. Rerun comparisons should log valid counts and failures explicitly.

Similarity/overlap scores do not establish factual correctness, valid JSON or fitness for patient care. Human clinical review and a suitable independent evaluation would be separate work, not a completed feature of this repository.
