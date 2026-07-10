"""Aggregate end-to-end pipeline statistics across the full Jakob corpus.

Walks `outputs/jakob_full_corpus/*/results.json` and the matching
`logs/jakob_full_corpus/<slug>.log`, calls `aggregate_run()` per lecture
(reusing scripts/eval_end_to_end.py), and writes a corpus-level summary to
`outputs/eval_end_to_end_jakob_full.json`.

The output JSON has two top-level blocks:

  - `per_lecture`: array of single-lecture aggregates (existing schema from
    eval_end_to_end.py), one entry per lecture.
  - `corpus`: aggregate stats across lectures:
      n_lectures, n_slides_total, total text/math detections, total
      math_rendered, weighted + arithmetic mean render-rate, std across
      lectures, min/max lecture render-rate, total drop counts, total /
      mean / median slide timing.

Usage:
  python scripts/aggregate_e2e_corpus.py
  python scripts/aggregate_e2e_corpus.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_end_to_end import aggregate_run  # noqa: E402

CORPUS_OUT = Path("outputs/jakob_full_corpus")
CORPUS_LOG = Path("logs/jakob_full_corpus")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--corpus-out", default=str(CORPUS_OUT))
    ap.add_argument("--corpus-log", default=str(CORPUS_LOG))
    args = ap.parse_args()

    corpus_out = Path(args.corpus_out)
    corpus_log = Path(args.corpus_log)

    lecture_dirs = sorted(p for p in corpus_out.iterdir() if p.is_dir())
    per_lecture = []
    skipped = []
    for ldir in lecture_dirs:
        results_json = ldir / "results.json"
        log_file = corpus_log / f"{ldir.name}.log"
        if not results_json.is_file():
            skipped.append({"slug": ldir.name, "reason": "no results.json"})
            continue
        if not log_file.is_file():
            skipped.append({"slug": ldir.name, "reason": "no log"})
            continue
        agg = aggregate_run(results_json, log_file, tag=ldir.name)
        per_lecture.append(agg)

    if not per_lecture:
        print(f"No lectures aggregated. Skipped: {skipped}", file=sys.stderr)
        return 2

    n_lectures = len(per_lecture)
    n_slides_total = sum(x["n_slides"] for x in per_lecture)
    text_total = sum(x["text_detections"] for x in per_lecture)
    math_total = sum(x["math_detections"] for x in per_lecture)
    rendered_total = sum(x["math_rendered"] for x in per_lecture)

    render_rates = [
        x["math_render_rate"]
        for x in per_lecture
        if x["math_render_rate"] is not None
    ]

    typeset_drops = sum(x["drops"]["typeset_filter_drops"] for x in per_lecture)
    dedupe_drops = sum(x["drops"]["post_ocr_dedupe_drops"] for x in per_lecture)
    garbage_drops = sum(x["drops"]["math_garbage_rejects"] for x in per_lecture)
    covered_drops = sum(
        x["drops"].get("text_covered_by_math_drops", 0) for x in per_lecture
    )

    def _dist(values):
        """Mean/std/median/min/max for a list of per-lecture numbers."""
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return {
            "mean": round(statistics.mean(vals), 2),
            "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": len(vals),
        }

    timings_total = sum(
        (x["timing"]["total_s"] if x["timing"] else 0.0) for x in per_lecture
    )
    timings_means = [x["timing"]["mean_s"] for x in per_lecture if x["timing"]]
    timings_medians = [x["timing"]["median_s"] for x in per_lecture if x["timing"]]
    # Pool every per-slide wall-clock across the corpus (not per-lecture means).
    pooled_slide_timings = [
        t
        for x in per_lecture
        if x["timing"]
        for t in x["timing"].get("per_slide_s", [])
    ]
    slides_with_timing = sum(
        (x["timing"]["n_slides_with_timing"] if x["timing"] else 0)
        for x in per_lecture
    )

    corpus = {
        "n_lectures": n_lectures,
        "n_slides_total": n_slides_total,
        "text_detections_total": text_total,
        "math_detections_total": math_total,
        "math_rendered_total": rendered_total,
        "math_render_rate_weighted": round(rendered_total / math_total, 4)
        if math_total
        else None,
        "math_render_rate_arith_mean": round(statistics.mean(render_rates), 4)
        if render_rates
        else None,
        "math_render_rate_std": round(statistics.stdev(render_rates), 4)
        if len(render_rates) > 1
        else None,
        "math_render_rate_median": round(statistics.median(render_rates), 4)
        if render_rates
        else None,
        # Several lectures can tie at a perfect 1.0 render rate; report the
        # count (and the list) so callers don't name a single arbitrary "best".
        "n_lectures_full_render": sum(1 for r in render_rates if r >= 1.0),
        "lectures_full_render": [
            x["tag"] for x in per_lecture if x["math_render_rate"] == 1.0
        ],
        "math_render_rate_min_lecture": min(
            (x for x in per_lecture if x["math_render_rate"] is not None),
            key=lambda x: x["math_render_rate"],
            default=None,
        ),
        "math_render_rate_max_lecture": max(
            (x for x in per_lecture if x["math_render_rate"] is not None),
            key=lambda x: x["math_render_rate"],
            default=None,
        ),
        "drops_total": {
            "typeset_filter_drops": typeset_drops,
            "post_ocr_dedupe_drops": dedupe_drops,
            "text_covered_by_math_drops": covered_drops,
            "math_garbage_rejects": garbage_drops,
        },
        "drops_per_lecture_dist": {
            "typeset_filter_drops": _dist(
                [x["drops"]["typeset_filter_drops"] for x in per_lecture]
            ),
            "post_ocr_dedupe_drops": _dist(
                [x["drops"]["post_ocr_dedupe_drops"] for x in per_lecture]
            ),
            "text_covered_by_math_drops": _dist(
                [x["drops"].get("text_covered_by_math_drops", 0) for x in per_lecture]
            ),
            "math_garbage_rejects": _dist(
                [x["drops"]["math_garbage_rejects"] for x in per_lecture]
            ),
        },
        "math_render_rate_dist": _dist(render_rates),
        "timing_per_lecture_median_dist": _dist(timings_medians),
        "timing_per_slide_pooled": {
            "n": len(pooled_slide_timings),
            "mean_s": round(statistics.mean(pooled_slide_timings), 1)
            if pooled_slide_timings
            else None,
            "median_s": round(statistics.median(pooled_slide_timings), 1)
            if pooled_slide_timings
            else None,
            "min_s": round(min(pooled_slide_timings), 1)
            if pooled_slide_timings
            else None,
            "max_s": round(max(pooled_slide_timings), 1)
            if pooled_slide_timings
            else None,
        },
        "timing_total_s": round(timings_total, 1),
        "timing_per_lecture_mean_s_avg": round(statistics.mean(timings_means), 1)
        if timings_means
        else None,
        "timing_per_lecture_median_s_avg": round(statistics.mean(timings_medians), 1)
        if timings_medians
        else None,
        "slides_with_timing_total": slides_with_timing,
    }

    if isinstance(corpus["math_render_rate_min_lecture"], dict):
        corpus["math_render_rate_min_lecture"] = {
            "slug": corpus["math_render_rate_min_lecture"]["tag"],
            "rate": corpus["math_render_rate_min_lecture"]["math_render_rate"],
            "math_detections": corpus["math_render_rate_min_lecture"]["math_detections"],
        }
    if isinstance(corpus["math_render_rate_max_lecture"], dict):
        corpus["math_render_rate_max_lecture"] = {
            "slug": corpus["math_render_rate_max_lecture"]["tag"],
            "rate": corpus["math_render_rate_max_lecture"]["math_render_rate"],
            "math_detections": corpus["math_render_rate_max_lecture"]["math_detections"],
        }

    out = {
        "corpus": corpus,
        "skipped": skipped,
        "per_lecture": per_lecture,
    }

    out_path = Path(args.output_dir) / "eval_end_to_end_jakob_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(corpus, indent=2))
    if skipped:
        print(f"NOTE: {len(skipped)} lecture(s) skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
