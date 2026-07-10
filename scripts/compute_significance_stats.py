#!/usr/bin/env python3
"""Bootstrap CIs and paired significance tests for the paper's OCR results.

Inputs (produced by evaluate/eval_math_ocr.py and evaluate/eval_text_5way.py):
  outputs/eval_math_ocr_jakob_5backends.json   (134 expressions x 5 backends)
  outputs/jakob_5way_eval_v2.json              (100 text lines x 5 systems)

All resampling is deterministic (seed 42, 10,000 resamples, percentile CIs).
Paired p-values come from a sign-flip permutation test on the per-item CER
differences (two-sided, 10,000 permutations).

Writes outputs/significance_stats.json and prints a summary.
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 10_000
SEED = 42


def mean(xs):
    return sum(xs) / len(xs)


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(
        mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    lo = stats[int(math.floor(alpha / 2 * n_boot))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return mean(values), lo, hi


def paired_test(a, b, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """CI on mean(a-b) via paired bootstrap + sign-flip permutation p-value."""
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    obs = mean(diffs)
    _, lo, hi = bootstrap_ci(diffs, n_boot, seed, alpha)
    rng = random.Random(seed)
    n = len(diffs)
    hits = sum(
        1
        for _ in range(n_boot)
        if abs(mean([d if rng.random() < 0.5 else -d for d in diffs])) >= abs(obs)
    )
    p = (hits + 1) / (n_boot + 1)
    return {"mean_diff": obs, "ci_lo": lo, "ci_hi": hi, "p_two_sided": p}


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def main():
    out = {"n_boot": N_BOOT, "seed": SEED, "math": {}, "text": {}}

    # ---------------- math eval (134 expressions, 5 backends) ----------------
    math_eval = json.loads(
        (ROOT / "outputs/eval_math_ocr_jakob_5backends.json").read_text()
    )
    backends = math_eval["backends"]

    # align every backend's per-sample list by image id
    def cer_by_image(bk, which):
        return {s["image"]: s[which]["cer"] for s in bk["per_sample"]}

    images = sorted(cer_by_image(backends["vlm"], "post"))
    for name, bk in backends.items():
        entry = {}
        for which in ("raw", "post"):
            by_img = cer_by_image(bk, which)
            assert sorted(by_img) == images, f"image mismatch in {name}"
            m, lo, hi = bootstrap_ci([by_img[i] for i in images])
            reported = bk["overall"][which]["cer"]
            assert abs(m - reported) < 1e-9, (
                f"{name}/{which}: per-item mean {m} != reported overall {reported}"
            )
            entry[which] = {"mean_cer": m, "ci_lo": lo, "ci_hi": hi}
        out["math"][name] = entry

    # ExpRate Wilson CI for the deployed backend
    k = sum(s["post"]["exp_rate"] for s in backends["vlm"]["per_sample"])
    p, lo, hi = wilson_ci(int(k), len(images))
    out["math"]["vlm_exprate_wilson"] = {
        "k": int(k), "n": len(images), "p": p, "ci_lo": lo, "ci_hi": hi
    }

    # paired: deployed VLM vs the closest competitor (post-filter CER)
    vlm = cer_by_image(backends["vlm"], "post")
    paddle = cer_by_image(backends["paddleocr-vl"], "post")
    out["math"]["paired_vlm_vs_paddle_post"] = paired_test(
        [paddle[i] for i in images], [vlm[i] for i in images]
    )

    # ---------------- text eval (100 lines, 5 systems) ----------------
    text_eval = json.loads((ROOT / "outputs/jakob_5way_eval_v2.json").read_text())
    systems = [
        "base_trocr_iam", "reptile_meta", "jakob_finetune",
        "vlm_qwen3", "vlm_qwen3_adapted",
    ]

    def text_cers(sysname):
        return {p["image"]: p["cer"] for p in text_eval[sysname]["predictions"]}

    line_ids = sorted(text_cers("vlm_qwen3"))
    per_sys = {}
    for s in systems:
        by_img = text_cers(s)
        assert sorted(by_img) == line_ids, f"line mismatch in {s}"
        vals = [by_img[i] for i in line_ids]
        assert abs(mean(vals) - text_eval[s]["mean_cer"]) < 1e-9
        m, lo, hi = bootstrap_ci(vals)
        out["text"][s] = {"mean_cer": m, "ci_lo": lo, "ci_hi": hi}
        per_sys[s] = vals

    for label, a, b in [
        ("finetune_vs_reptile", "reptile_meta", "jakob_finetune"),
        ("vlm_zeroshot_vs_finetune", "jakob_finetune", "vlm_qwen3"),
        ("adapter_vs_zeroshot", "vlm_qwen3", "vlm_qwen3_adapted"),
    ]:
        out["text"][f"paired_{label}"] = paired_test(per_sys[a], per_sys[b])

    dest = ROOT / "outputs/significance_stats.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}\n")

    pc = lambda x: f"{100*x:.2f}%"
    print("MATH (post-filter, mean per-item CER, 95% bootstrap CI):")
    for name in ("vlm", "paddleocr-vl", "trocr", "pix2tex", "tamer"):
        e = out["math"][name]["post"]
        print(f"  {name:14s} {pc(e['mean_cer'])}  [{pc(e['ci_lo'])}, {pc(e['ci_hi'])}]")
    w = out["math"]["vlm_exprate_wilson"]
    print(f"  vlm ExpRate {w['k']}/{w['n']} = {pc(w['p'])}  Wilson [{pc(w['ci_lo'])}, {pc(w['ci_hi'])}]")
    t = out["math"]["paired_vlm_vs_paddle_post"]
    print(f"  paddle - vlm: {pc(t['mean_diff'])}  [{pc(t['ci_lo'])}, {pc(t['ci_hi'])}]  p={t['p_two_sided']:.5f}")

    print("\nTEXT (mean per-item CER, 95% bootstrap CI):")
    for s in systems:
        e = out["text"][s]
        print(f"  {s:18s} {pc(e['mean_cer'])}  [{pc(e['ci_lo'])}, {pc(e['ci_hi'])}]")
    for label in ("finetune_vs_reptile", "vlm_zeroshot_vs_finetune", "adapter_vs_zeroshot"):
        t = out["text"][f"paired_{label}"]
        print(f"  {label:26s} diff {pc(t['mean_diff'])}  [{pc(t['ci_lo'])}, {pc(t['ci_hi'])}]  p={t['p_two_sided']:.5f}")


if __name__ == "__main__":
    main()
