# Evaluation datasets

These are the two evaluation sets released with the paper *"Typeset
Replacement of Handwritten Text and Mathematics on Lecture Slides Using
Vision-Language Models"* (IEEE Access). All crops originate from
Prof. Dr. Judith Jakob's handwritten annotations on her lecture slides
and are released with her permission for research use.

## Handwritten-mathematics evaluation set (134 expressions)

The set the paper's math-OCR tables report on (n = 134). It is stored as
two directories that are always evaluated together:

| Directory | Crops | Ground truth |
|---|---|---|
| `jakob_math_eval/` | 104 (`eq_*.jpg`) | `samples.json`, key `gt_latex` |
| `jakob_math/` | 30 (`math_*.jpg`) | `transcriptions.json`, key `latex` |

Reproduce the paper's evaluation with:

```bash
python evaluate/eval_math_ocr.py \
    --data data/jakob_math_eval/samples.json \
    --data data/jakob_math/transcriptions.json \
    --backends vlm,tamer,pix2tex,trocr,paddleocr-vl
```

Scoring is exact match (ExpRate) after canonicalisation, character error
rate, and corpus BLEU; see the paper's Experiments section for the
protocol and `evaluate/eval_math_ocr.py` for the implementation.

## Handwritten German text evaluation set (100 lines)

The set behind the paper's five-backend text-OCR comparison (n = 100,
character error rate). Disjoint from the QLoRA adaptation train/val data.

- `jakob_text_v2/images/` — 100 line crops (`text_*.jpg`)
- `jakob_text_v2/samples.json` — a JSON list of
  `{"image": "jakob_text_v2/images/text_NNN.jpg", "text": "<ground truth>"}`;
  image paths resolve relative to `data/`.

## Annotation protocol

All bounding boxes and transcriptions were produced by the paper's first
author (boxes in LabelImg); single-annotator labelling is stated as a
limitation in the paper's *Reproducibility and Annotation Protocol*
section. If you find a transcription error, please open an issue.
