# Experiment methods

## Dataset and split

The notebook downloads `AGBonnet/augmented-clinical-notes` through Hugging Face Datasets and uses the `note` and `summary` fields. The recorded dataset has 30,000 rows. Whitespace is normalized, missing/very short fields are removed, and 5,000 rows are sampled with seed 42. A reproducible row-level split yields 3,500 training, 750 validation and 750 test rows. The first 50 test rows form the metric comparison.

This is a row split, not a verified patient/case-level split. The implementation does not check duplicated or related case records across splits. Reference targets are JSON-style structured text; they are not necessarily shorter than the source notes. The original EDA output reports median note/target lengths of 316/446 words and median target-to-note word ratio 1.407. Structured transformation is a more accurate description than guaranteed compression.

## Three approaches

1. TF-IDF extractive routine: splits sentences with a regex, discards sentences of four words or fewer, scores sentence vectors against their mean and chooses up to five in original order. The original code falls back to the first five eligible sentences on any ranking error; the uploaded run does not log whether this happened. This edition converts the mean vector to a NumPy array to avoid the scikit-learn `np.matrix` incompatibility.
2. Pretrained BART: `facebook/bart-large-cnn`, without project-specific fine-tuning. The notebook's generation call uses only the first 1,024 characters and a 256-token generation limit.
3. Fine-tuned BART: the same starting checkpoint, trained on clinical-note/target pairs. Training input/target limits are 512/384 tokens. Evaluation passes the note to the pipeline with tokenizer truncation and uses a 384-token generation limit, four beams, length penalty 1.2 and no-repeat 3-grams.

The methods have unequal input/decoding limits. Therefore the recorded comparison does not isolate the effect of fine-tuning alone. No JSON parser or output schema validator is applied; the saved generated example contains malformed JSON-style text.

## Training

Learning rate 3e-5; cosine scheduler; nominal warmup fraction 6%; weight decay 0.01; label smoothing 0.1; up to five epochs. Validation loss chooses the best saved checkpoint; early-stopping patience is two epochs. FP16 is requested when CUDA is available.

Per-device batch size is two, with four accumulation steps. Effective global batch is nominally eight times the distributed device count. The original manual warmup estimate uses a single-device floor calculation, so it is an approximation when training topology changes.

## Lightweight source utilities

`src/heuristics.py` exposes cleaning, TF-IDF selection, regex extraction, source-token overlap and custom lexicon concept F1 for inspection/testing. The notebook remains the main training/inference workflow. No web interface or pretrained project checkpoint is bundled.
