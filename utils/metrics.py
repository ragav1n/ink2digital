"""
Evaluation metrics for OCR and detection.
Supports CER, WER, mAP, and BLEU for the German Lecture Slide OCR project.
"""

import re
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Character Error Rate / Word Error Rate
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = edit_distance(hyp, ref) / len(ref).
    Returns value in [0, inf). Lower is better.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = _edit_distance(hypothesis, reference)
    return dist / len(reference)


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate = edit_distance(hyp_words, ref_words) / len(ref_words).
    Returns value in [0, inf). Lower is better.
    """
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = _edit_distance(" ".join(hyp_words), " ".join(ref_words))
    # Use word-level edit distance
    dp = list(range(len(ref_words) + 1))
    for i in range(1, len(hyp_words) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(ref_words) + 1):
            temp = dp[j]
            dp[j] = prev if hyp_words[i-1] == ref_words[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[len(ref_words)] / len(ref_words)


def batch_cer(hypotheses: List[str], references: List[str]) -> dict:
    """Compute mean CER over a batch. Returns dict with mean and per-sample."""
    assert len(hypotheses) == len(references)
    cers = [compute_cer(h, r) for h, r in zip(hypotheses, references)]
    return {
        "mean_cer": float(np.mean(cers)),
        "per_sample": cers,
        "n_samples": len(cers),
    }


def batch_wer(hypotheses: List[str], references: List[str]) -> dict:
    """Compute mean WER over a batch."""
    assert len(hypotheses) == len(references)
    wers = [compute_wer(h, r) for h, r in zip(hypotheses, references)]
    return {
        "mean_wer": float(np.mean(wers)),
        "per_sample": wers,
        "n_samples": len(wers),
    }


# ---------------------------------------------------------------------------
# Detection metrics: IoU and mAP
# ---------------------------------------------------------------------------

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precisions[recalls >= thr]
        ap += prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
    return ap / 11.0


def compute_map(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute mAP@IoU_threshold.

    Args:
        predictions: List of {'boxes': [[x1,y1,x2,y2],...], 'labels': [...], 'scores': [...]}
        ground_truths: List of {'boxes': [[x1,y1,x2,y2],...], 'labels': [...]}
        iou_threshold: IoU threshold for a true positive match.
        class_names: Optional list of class name strings.

    Returns:
        dict with 'mAP', 'per_class_AP', and optionally class names.
    """
    if class_names is None:
        # Infer number of classes
        all_labels = []
        for gt in ground_truths:
            all_labels.extend(gt["labels"])
        n_classes = max(all_labels) + 1 if all_labels else 0
        class_names = [str(i) for i in range(n_classes)]

    n_classes = len(class_names)
    per_class_ap = {}

    for cls_idx, cls_name in enumerate(class_names):
        # Collect all predictions for this class across all images
        tp_list, fp_list, scores_list = [], [], []
        n_gt_total = 0

        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == cls_idx]
            pred_boxes = [(b, s) for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]) if l == cls_idx]
            # Sort by descending score
            pred_boxes.sort(key=lambda x: -x[1])

            n_gt_total += len(gt_boxes)
            matched = [False] * len(gt_boxes)

            for box, score in pred_boxes:
                scores_list.append(score)
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched[best_j] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if n_gt_total == 0:
            per_class_ap[cls_name] = 0.0
            continue

        # Sort by score
        order = np.argsort([-s for s in scores_list])
        tp_arr = np.array(tp_list)[order]
        fp_arr = np.array(fp_list)[order]

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        recalls = tp_cum / n_gt_total
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        per_class_ap[cls_name] = compute_ap(recalls, precisions)

    map_val = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {
        "mAP": map_val,
        "per_class_AP": per_class_ap,
        "iou_threshold": iou_threshold,
    }


# ---------------------------------------------------------------------------
# BLEU (for math expression evaluation)
# ---------------------------------------------------------------------------

def compute_bleu(hypothesis: str, reference: str, n: int = 4) -> float:
    """
    Simple corpus BLEU score (single sentence pair).
    Returns value in [0, 1]. Higher is better.
    """
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    log_score = 0.0
    for k in range(1, n + 1):
        hyp_ngrams: dict = {}
        ref_ngrams: dict = {}

        for i in range(len(hyp_tokens) - k + 1):
            ng = tuple(hyp_tokens[i:i+k])
            hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

        for i in range(len(ref_tokens) - k + 1):
            ng = tuple(ref_tokens[i:i+k])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        clip_count = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items())
        total_count = max(1, sum(hyp_ngrams.values()))
        precision = clip_count / total_count
        log_score += np.log(precision + 1e-10) / n

    return float(bp * np.exp(log_score))


if __name__ == "__main__":
    # Quick sanity checks
    print("CER test:", compute_cer("Hallo Welt", "Hallo Welt"))   # 0.0
    print("CER test:", compute_cer("Halo Welt", "Hallo Welt"))    # ~0.1
    print("WER test:", compute_wer("Hallo Welt", "Hallo Welt"))   # 0.0
    print("IoU test:", compute_iou([0,0,10,10], [5,5,15,15]))     # 25/175
    print("BLEU test:", compute_bleu("the cat sat", "the cat sat on the mat"))
