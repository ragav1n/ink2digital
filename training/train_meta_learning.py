"""
Meta-learning training script for professor-specific OCR adaptation.
Phase 3 (Week 7-9): Train MAML on IAM German writer tasks.

This script:
1. Loads the fine-tuned TrOCR (Phase 2 output) as the base model
2. Wraps it with MAML
3. Meta-trains on IAM German writer tasks
4. Evaluates 5-shot professor adaptation

Architecture:
    Base:  Fine-tuned TrOCR (checkpoints/trocr_german/best) — Phase 2
    Meta:  MAML outer-loop with writer-level tasks
    Goal:  2-3% CER after 5-shot adaptation (vs. 5-8% fine-tuned, 15-20% baseline)

Usage:
    python training/train_meta_learning.py \
        --train-data data/processed/german_text/german_text_train.json \
        --val-data data/processed/german_text/german_text_val.json \
        --base-model checkpoints/trocr_german/best \
        --output-dir checkpoints/meta_learning \
        --epochs 50

    # With learn2learn (recommended):
    pip install learn2learn
    python training/train_meta_learning.py --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from models.meta_learning_ocr import MAMLOCRWrapper, create_writer_tasks


def train_meta_learning(
    train_manifest: Path,
    val_manifest: Path,
    base_model_path: str = 'checkpoints/trocr_german/best',
    output_dir: Path = Path('checkpoints/meta_learning'),
    num_epochs: int = 50,
    tasks_per_epoch: int = 100,
    batch_tasks: int = 8,
    n_support: int = 10,
    n_query: int = 10,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    device: str = 'cuda',
) -> dict:
    """
    Run MAML meta-training for professor-specific OCR adaptation.

    Args:
        train_manifest: Path to training JSON manifest (IAM German).
        val_manifest: Path to validation JSON manifest.
        base_model_path: Fine-tuned TrOCR from Phase 2.
        output_dir: Where to save MAML checkpoints.
        num_epochs: Number of meta-training epochs.
        tasks_per_epoch: Writer tasks sampled per epoch.
        batch_tasks: Tasks per meta-gradient update.
        n_support: Support samples per task.
        n_query: Query samples per task.
        inner_lr: MAML inner loop learning rate.
        outer_lr: MAML outer loop learning rate.
        inner_steps: Inner loop gradient steps.
        device: Compute device.

    Returns:
        Dict with best_meta_cer and epochs_trained.
    """
    logger.info("=" * 60)
    logger.info("MAML Meta-Learning Training (Phase 3)")
    logger.info("=" * 60)
    logger.info(f"Base model:      {base_model_path}")
    logger.info(f"Train manifest:  {train_manifest}")
    logger.info(f"Val manifest:    {val_manifest}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Epochs:          {num_epochs}")
    logger.info(f"Tasks/epoch:     {tasks_per_epoch}")
    logger.info(f"Inner LR:        {inner_lr}")
    logger.info(f"Outer LR:        {outer_lr}")
    logger.info(f"Inner steps:     {inner_steps}")
    logger.info(f"Support/query:   {n_support}/{n_query}")
    logger.info("=" * 60)

    # Build writer tasks
    logger.info("Building writer tasks from training manifest...")
    train_tasks = create_writer_tasks(
        train_manifest,
        n_support=n_support,
        n_query=n_query,
        min_samples_per_writer=n_support + n_query,
    )
    val_tasks = create_writer_tasks(
        val_manifest,
        n_support=n_support,
        n_query=n_query,
        min_samples_per_writer=n_support + n_query,
    )

    if not train_tasks:
        logger.error(
            "No writer tasks created. Check that training manifest has writer_id fields "
            "and sufficient samples per writer."
        )
        return {}

    logger.info(f"Training tasks:   {len(train_tasks)} writers")
    logger.info(f"Validation tasks: {len(val_tasks)} writers")

    # Initialize MAML wrapper
    logger.info(f"Initializing MAML wrapper with base model: {base_model_path}")
    wrapper = MAMLOCRWrapper(
        base_model_path=base_model_path,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        device=device,
    )

    # Meta-train
    t0 = time.time()
    results = wrapper.meta_train(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        num_epochs=num_epochs,
        tasks_per_epoch=tasks_per_epoch,
        batch_tasks=batch_tasks,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Meta-training complete!")
    logger.info(f"  Best meta CER:  {results.get('best_meta_cer', 'N/A'):.4f} "
                f"({results.get('best_meta_cer', 0)*100:.2f}%)")
    logger.info(f"  Epochs trained: {results.get('epochs', num_epochs)}")
    logger.info(f"  Total time:     {elapsed/60:.1f} min")
    logger.info(f"  Checkpoint:     {output_dir}/meta_checkpoint_best.pt")
    logger.info("=" * 60)

    return results


def evaluate_adaptation(
    test_manifest: Path,
    meta_checkpoint: Path,
    n_shot: int = 5,
    n_eval_writers: int = 20,
    device: str = 'cuda',
) -> dict:
    """
    Evaluate few-shot adaptation on held-out writers.

    Simulates the professor adaptation scenario:
    - Give model n_shot examples from a new writer
    - Measure CER on unseen examples from same writer
    - Compare: before adaptation vs. after adaptation

    Args:
        test_manifest: Path to test manifest.
        meta_checkpoint: Path to meta-learned MAML checkpoint.
        n_shot: Number of support samples for adaptation.
        n_eval_writers: Number of writers to evaluate.
        device: Compute device.

    Returns:
        Dict with CER before/after adaptation.
    """
    import torch
    import numpy as np
    from utils.metrics import compute_cer
    from utils.image_utils import load_image
    from PIL import Image as PILImage

    logger.info(f"\nEvaluating {n_shot}-shot adaptation on {n_eval_writers} writers...")

    # Load meta-learned model
    wrapper = MAMLOCRWrapper(device=device)
    if meta_checkpoint.exists():
        ckpt = torch.load(str(meta_checkpoint), map_location=device)
        wrapper.meta_model.load_state_dict(ckpt['meta_model_state'])
        logger.info(f"Loaded meta checkpoint: {meta_checkpoint}")

    # Create test tasks (new writers not seen during meta-training)
    test_tasks = create_writer_tasks(
        test_manifest,
        n_support=n_shot,
        n_query=10,
        min_samples_per_writer=n_shot + 5,
    )[:n_eval_writers]

    if not test_tasks:
        logger.warning("No test tasks available for adaptation evaluation")
        return {}

    before_cers = []
    after_cers = []

    for i, task in enumerate(test_tasks):
        writer_id = task['writer_id']
        support = task['support'][:n_shot]
        query = task['query']

        # Load query images
        query_images = wrapper._load_images([q['image'] for q in query])
        query_texts = [q['text'] for q in query]

        # Predict BEFORE adaptation (use meta-learned init as-is)
        with torch.no_grad():
            pv = wrapper.processor(
                images=query_images, return_tensors='pt'
            ).pixel_values.to(wrapper.device)
            ids = wrapper.meta_model.generate(pv, max_new_tokens=128)
            hyps_before = wrapper.processor.batch_decode(ids, skip_special_tokens=True)

        # Adapt to this writer
        wrapper.adapt(support, steps=n_shot)

        # Predict AFTER adaptation
        with torch.no_grad():
            pv = wrapper.processor(
                images=query_images, return_tensors='pt'
            ).pixel_values.to(wrapper.device)
            ids = wrapper._adapted_learner.generate(pv, max_new_tokens=128)
            hyps_after = wrapper.processor.batch_decode(ids, skip_special_tokens=True)

        cer_before = np.mean([compute_cer(h, r) for h, r in zip(hyps_before, query_texts)])
        cer_after = np.mean([compute_cer(h, r) for h, r in zip(hyps_after, query_texts)])

        before_cers.append(cer_before)
        after_cers.append(cer_after)

        logger.info(
            f"Writer {writer_id}: "
            f"Before={cer_before:.4f} ({cer_before*100:.2f}%) → "
            f"After={cer_after:.4f} ({cer_after*100:.2f}%)"
        )

    mean_before = float(np.mean(before_cers))
    mean_after = float(np.mean(after_cers))
    improvement = mean_before - mean_after

    results = {
        'n_shot': n_shot,
        'n_writers': len(test_tasks),
        'CER_before': mean_before,
        'CER_after': mean_after,
        'CER_improvement': improvement,
        'CER_improvement_pct': improvement / max(mean_before, 1e-6) * 100,
    }

    logger.info("\n" + "=" * 50)
    logger.info(f"Few-shot Adaptation Results ({n_shot}-shot)")
    logger.info(f"  Writers evaluated: {len(test_tasks)}")
    logger.info(f"  CER before: {mean_before:.4f} ({mean_before*100:.2f}%)")
    logger.info(f"  CER after:  {mean_after:.4f} ({mean_after*100:.2f}%)")
    logger.info(f"  Improvement: {improvement*100:.2f}% absolute "
                f"({improvement/max(mean_before, 1e-6)*100:.1f}% relative)")
    logger.info("=" * 50)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-learning training (MAML Phase 3)")
    parser.add_argument('--train-data', type=Path,
                        default=Path('data/processed/german_text/german_text_train.json'))
    parser.add_argument('--val-data', type=Path,
                        default=Path('data/processed/german_text/german_text_val.json'))
    parser.add_argument('--test-data', type=Path,
                        default=Path('data/processed/german_text/german_text_test.json'))
    parser.add_argument('--base-model', type=str,
                        default='checkpoints/trocr_german/best',
                        help='Fine-tuned TrOCR checkpoint from Phase 2')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('checkpoints/meta_learning'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tasks-per-epoch', type=int, default=100)
    parser.add_argument('--batch-tasks', type=int, default=8)
    parser.add_argument('--n-support', type=int, default=10)
    parser.add_argument('--n-query', type=int, default=10)
    parser.add_argument('--inner-lr', type=float, default=0.01)
    parser.add_argument('--outer-lr', type=float, default=0.001)
    parser.add_argument('--inner-steps', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, only evaluate adaptation')
    parser.add_argument('--n-shot', type=int, default=5,
                        help='Shots for adaptation evaluation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.eval_only:
        results = train_meta_learning(
            train_manifest=args.train_data,
            val_manifest=args.val_data,
            base_model_path=args.base_model,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            tasks_per_epoch=args.tasks_per_epoch,
            batch_tasks=args.batch_tasks,
            n_support=args.n_support,
            n_query=args.n_query,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            inner_steps=args.inner_steps,
            device=args.device,
        )

    # Evaluate few-shot adaptation
    meta_ckpt = args.output_dir / 'meta_checkpoint_best.pt'
    if meta_ckpt.exists():
        eval_results = evaluate_adaptation(
            test_manifest=args.test_data,
            meta_checkpoint=meta_ckpt,
            n_shot=args.n_shot,
            device=args.device,
        )

        # Save eval results
        out_path = Path('outputs/eval_meta_learning.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Eval results saved -> {out_path}")
    else:
        logger.warning(f"No meta checkpoint found at {meta_ckpt}. Run training first.")
