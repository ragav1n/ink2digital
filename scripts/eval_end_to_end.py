"""Aggregate end-to-end pipeline statistics from a previous infer.py run.

Inputs:
  - <output-dir>/results.json   (per-slide detection records, with 'rendered' flag)
  - <log-file>                  (per-slide wall-clock timing + filter drop counts)

Output:
  outputs/eval_end_to_end_<tag>.json with:
    - n_slides, text/math detection counts, math render rate
    - typeset-filter / dedupe / garbage-filter drop totals
    - timing: mean / median / min / max / total (over slides that have a log line)

Usage:
  python scripts/eval_end_to_end.py \
      --results-json outputs/jakob_v10_latex/results.json \
      --log-file logs/jakob_v10_latex.log \
      --tag jakob_v10_latex
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def aggregate_run(results_json: Path, log_file: Path, tag: str | None = None) -> dict:
    """Compute the end-to-end summary for a single infer.py run.

    Returns the dict that the CLI would otherwise write to disk. The
    `per_slide` array is included so callers (e.g. corpus-level aggregator)
    can drill in if needed.
    """
    results = json.loads(Path(results_json).read_text())
    log = Path(log_file).read_text()

    text_total = math_total = math_rendered = 0
    per_slide = []
    for k, dets in results.items():
        t = m = r = 0
        for d in dets:
            cls = d.get("type") or d.get("class", "")
            if cls == "text":
                t += 1
            elif cls == "math":
                m += 1
                if d.get("rendered"):
                    r += 1
        per_slide.append({"slide": Path(k).name, "text": t, "math": m, "math_rendered": r})
        text_total += t
        math_total += m
        math_rendered += r

    timings = [float(x) for x in re.findall(r"Saved -> .+?\(([\d.]+)s\)", log)]
    timing_stats = None
    if timings:
        timing_stats = {
            "n_slides_with_timing": len(timings),
            "mean_s": round(statistics.mean(timings), 1),
            "median_s": round(statistics.median(timings), 1),
            "min_s": round(min(timings), 1),
            "max_s": round(max(timings), 1),
            "total_s": round(sum(timings), 1),
            # Raw per-slide values so a corpus aggregator can pool them.
            "per_slide_s": [round(t, 1) for t in timings],
        }

    drops = {
        "typeset_filter_drops": sum(int(m) for m in re.findall(r"Typeset filter dropped (\d+) region", log)),
        "post_ocr_dedupe_drops": sum(int(m) for m in re.findall(r"Post-OCR dedupe: dropped (\d+) near-duplicate", log)),
        # One log line per overlay dropped because a typeset-math render already
        # owns that area (infer.py _dedupe_text_results, second pass).
        "text_covered_by_math_drops": len(re.findall(r"Dropped text overlay covered by typeset math", log)),
        # NOTE: this is emitted at logger.debug level in infer.py; it is only
        # accurate when the run captured debug logs. The render rate above is
        # independent of log level (it reads the per-region 'rendered' flag).
        "math_garbage_rejects": len(re.findall(r"Math OCR rejected garbage:", log)),
    }

    return {
        "tag": tag,
        "results_json": str(results_json),
        "log_file": str(log_file),
        "n_slides": len(results),
        "text_detections": text_total,
        "math_detections": math_total,
        "math_rendered": math_rendered,
        "math_render_rate": round(math_rendered / math_total, 4) if math_total else None,
        "drops": drops,
        "timing": timing_stats,
        "per_slide": per_slide,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    out = aggregate_run(Path(args.results_json), Path(args.log_file), tag=args.tag)

    out_path = Path(args.output_dir) / f"eval_end_to_end_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_slide"}, indent=2))


if __name__ == "__main__":
    main()
