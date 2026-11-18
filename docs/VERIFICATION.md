# Verification

Checked during preparation of this repository edition:

- Eight utility tests passed: cleaning, source-sentence selection, parameter checks, lexical overlap, regex extraction, concept F1 and documented heuristic failure cases.
- All 25 notebook code cells parse successfully. Saved code outputs and execution counts are cleared; notebook IDs and kernel metadata are normalized.
- All 24 aggregate metric values were extracted from the supplied notebook's recorded output. Public labels identify the actual heuristic methods. No per-example scores or trained checkpoint were supplied for independent recomputation.
- The TF-IDF mean vector is converted from `np.matrix` to a NumPy array for cosine similarity compatibility. The experiment settings and historical input-budget asymmetry are retained and documented.
- Dependency installation is moved to requirements files instead of automatically downgrading a hosted notebook environment.
- The conclusion no longer claims improvement on every metric, low hallucination or validated clinical quality.
- No clinical-note examples, dataset records, generated summaries or model weights are included. Original output cells are cleared.
- Local Markdown links and source syntax were checked; common credential/private-path pattern scanning found no matching secrets.

Full dependency installation, model downloads, supervised training, generation and model-based metric computation were not rerun. The recorded metrics are historical aggregate outputs. The original narrative paper is replaced with the code-based methods, metric definitions and results documentation in this folder; its unsupported clinical conclusions are not republished.
