"""
MAML (Model-Agnostic Meta-Learning) wrapper for professor-specific OCR adaptation.
Phase 3 (Week 7-9): Enable few-shot adaptation to a specific professor's handwriting.

MAML learns an initialization θ* such that after K gradient steps on a small set
of professor samples (the support set), the model achieves low CER on that professor's
handwriting. Target: ~2-3% CER after 5-shot adaptation.

Architecture:
    Base model:  Fine-tuned TrOCR (from Phase 2) - encoder-decoder OCR
    Meta-wrapper: MAML outer/inner loops using learn2learn
    Task: Each 'task' = one writer's handwriting style
    Support set: 5-10 samples from a professor
    Query set:   Unseen samples from the same professor

References:
    - MAML: Finn et al. (2017) "Model-Agnostic Meta-Learning"
    - learn2learn: https://github.com/learnables/learn2learn

Usage:
    from models.meta_learning_ocr import MAMLOCRWrapper, create_writer_tasks

    wrapper = MAMLOCRWrapper('checkpoints/trocr_german/best')
    wrapper.meta_train(train_tasks, val_tasks)

    # Professor adaptation (5-shot):
    wrapper.adapt(professor_samples, steps=5)
    predictions = wrapper.predict(new_slide_crops)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

def create_writer_tasks(
    manifest_path: Path,
    n_support: int = 10,
    n_query: int = 10,
    min_samples_per_writer: int = 20,
) -> List[Dict]:
    """
    Create episode tasks from IAM German data.

    Each task represents one writer (professor analogue).
    Returns list of {'support': [(img_path, text),...], 'query': [...]}

    Args:
        manifest_path: Path to German text manifest (JSON).
        n_support: Number of support samples per task.
        n_query: Number of query samples per task.
        min_samples_per_writer: Minimum samples required to include a writer.
    """
    with open(manifest_path, encoding='utf-8') as f:
        data = json.load(f)

    # Group samples by writer_id
    by_writer: Dict[str, List[Dict]] = {}
    for sample in data['samples']:
        wid = sample.get('writer_id', 'unknown')
        by_writer.setdefault(wid, []).append(sample)

    tasks = []
    for wid, samples in by_writer.items():
        if len(samples) < min_samples_per_writer:
            continue

        random.shuffle(samples)
        support = samples[:n_support]
        query = samples[n_support:n_support + n_query]

        if len(query) < n_query // 2:
            continue

        tasks.append({
            'writer_id': wid,
            'support': support,
            'query': query,
        })

    logger.info(f"Created {len(tasks)} writer tasks from {manifest_path}")
    return tasks


# ---------------------------------------------------------------------------
# MAML wrapper
# ---------------------------------------------------------------------------

class MAMLOCRWrapper:
    """
    MAML wrapper for TrOCR enabling few-shot adaptation to new handwriting styles.

    The base model (TrOCR) is wrapped with MAML meta-learning, which trains
    a good weight initialization that can be quickly adapted to new writers.

    Adaptation workflow:
        1. Provide 5-10 sample images from a professor
        2. Run inner_loop_steps gradient updates
        3. Model is now specialized for that professor's style
        4. Run inference on new slide images

    Args:
        base_model_path: Path to fine-tuned TrOCR checkpoint (Phase 2 output).
        inner_lr: Learning rate for inner loop (task-specific updates).
        outer_lr: Learning rate for outer loop (meta-gradient updates).
        inner_steps: Number of gradient steps for adaptation.
        device: Compute device.
    """

    def __init__(
        self,
        base_model_path: str = 'checkpoints/trocr_german/best',
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        device: str = 'cuda',
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.model, self.processor = self._load_base_model(base_model_path)
        self.meta_model = self._wrap_with_maml()

    def _load_base_model(self, model_path: str):
        """Load the fine-tuned TrOCR model."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        path = Path(model_path)
        if path.exists():
            logger.info(f"Loading fine-tuned model from {model_path}")
            processor = TrOCRProcessor.from_pretrained(str(path))
            model = VisionEncoderDecoderModel.from_pretrained(str(path))
        else:
            logger.warning(f"Fine-tuned model not found at {model_path}")
            logger.warning("Falling back to microsoft/trocr-large-handwritten")
            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

        model.to(self.device)
        return model, processor

    def _wrap_with_maml(self):
        """Wrap model with MAML using learn2learn library."""
        try:
            import learn2learn as l2l
            meta_model = l2l.algorithms.MAML(
                self.model,
                lr=self.inner_lr,
                first_order=False,  # Use second-order MAML for best performance
            )
            logger.info("MAML wrapper created with learn2learn")
            return meta_model
        except ImportError:
            logger.warning("learn2learn not installed. Install: pip install learn2learn")
            logger.warning("Using manual MAML implementation as fallback")
            return ManualMAML(self.model, lr=self.inner_lr, steps=self.inner_steps)

    def _compute_loss(self, model, images, texts) -> torch.Tensor:
        """Compute CER-based loss for MAML inner loop."""
        pixel_values = self.processor(
            images=images, return_tensors='pt'
        ).pixel_values.to(self.device)

        labels = self.processor.tokenizer(
            texts, return_tensors='pt', max_length=128,
            padding='max_length', truncation=True
        ).input_ids.to(self.device)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        outputs = model(pixel_values=pixel_values, labels=labels)
        return outputs.loss

    def meta_train(
        self,
        train_tasks: List[Dict],
        val_tasks: List[Dict],
        num_epochs: int = 50,
        tasks_per_epoch: int = 100,
        batch_tasks: int = 8,
        output_dir: Path = Path('checkpoints/meta_learning'),
    ) -> dict:
        """
        Run MAML meta-training.

        Args:
            train_tasks: List of writer task dicts from create_writer_tasks().
            val_tasks: Validation tasks for CER evaluation.
            num_epochs: Number of meta-training epochs.
            tasks_per_epoch: Tasks sampled per epoch.
            batch_tasks: Tasks per meta-gradient update.
            output_dir: Where to save checkpoints.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_meta_cer = float('inf')
        training_log = []

        logger.info(f"Starting Reptile meta-training: {num_epochs} epochs, "
                    f"{len(train_tasks)} training tasks")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_tasks = 0

            # Sample tasks for this epoch
            sampled_tasks = random.sample(train_tasks, min(tasks_per_epoch, len(train_tasks)))

            for task in sampled_tasks:
                # Clone model for inner loop (one at a time to save VRAM)
                learner = self.meta_model.clone()

                support = task['support']
                support_images = self._load_images([s['image'] for s in support])
                support_texts = [s['text'] for s in support]

                # Inner loop: adapt to support set
                task_loss = 0.0
                for _ in range(self.inner_steps):
                    support_loss = self._compute_loss(learner, support_images, support_texts)
                    learner.adapt(support_loss)
                    task_loss += float(support_loss)

                # Reptile meta-update: move meta_model toward adapted parameters
                with torch.no_grad():
                    for p_meta, p_adapted in zip(
                        self.meta_model.parameters(), learner.parameters()
                    ):
                        p_meta.data += self.outer_lr * (p_adapted.data - p_meta.data)

                epoch_loss += task_loss / max(self.inner_steps, 1)
                n_tasks += 1

                # Free GPU memory immediately
                del learner
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(n_tasks, 1)

            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                val_cer = self._meta_evaluate(val_tasks[:20])
                is_best = val_cer < best_meta_cer
                if is_best:
                    best_meta_cer = val_cer
                    self._save_meta_checkpoint(epoch, val_cer, output_dir)

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Meta-Loss: {avg_loss:.4f} | "
                    f"Val CER: {val_cer:.4f} ({val_cer*100:.2f}%) | "
                    f"{'⭐ Best' if is_best else ''}"
                )
                training_log.append({
                    'epoch': epoch + 1,
                    'meta_loss': avg_loss,
                    'val_cer': val_cer,
                })

        with open(output_dir / 'meta_training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        return {'best_meta_cer': best_meta_cer, 'epochs': num_epochs}

    def adapt(
        self,
        support_samples: List[Dict],
        steps: Optional[int] = None,
    ) -> 'MAMLOCRWrapper':
        """
        Adapt model to a specific professor's handwriting.

        Args:
            support_samples: List of {'image': path, 'text': str} for the professor.
            steps: Number of adaptation steps (default: self.inner_steps).

        Returns:
            Self (with adapted weights, restorable).
        """
        steps = steps or self.inner_steps
        logger.info(f"Adapting to professor: {len(support_samples)} samples, {steps} steps")

        images = self._load_images([s['image'] for s in support_samples])
        texts = [s['text'] for s in support_samples]

        # Create a task-specific clone for adaptation
        self._adapted_learner = self.meta_model.clone()

        for step in range(steps):
            loss = self._compute_loss(self._adapted_learner, images, texts)
            self._adapted_learner.adapt(loss)
            logger.debug(f"  Adaptation step {step+1}: loss={float(loss):.4f}")

        logger.info("Professor adaptation complete")
        return self

    @torch.no_grad()
    def predict(
        self,
        images: List[np.ndarray],
        use_adapted: bool = True,
        postprocess_german: bool = True,
    ) -> List[str]:
        """
        Predict text for image crops.

        Args:
            images: List of RGB numpy arrays.
            use_adapted: Use professor-adapted model if available.
            postprocess_german: Apply German OCR post-processing.

        Returns:
            List of predicted text strings.
        """
        from PIL import Image as PILImage
        from utils.german_postprocessing import batch_correct

        model = getattr(self, '_adapted_learner', self.meta_model) if use_adapted else self.meta_model
        model.eval()

        pil_images = [PILImage.fromarray(img) for img in images]
        pixel_values = self.processor(
            images=pil_images, return_tensors='pt'
        ).pixel_values.to(self.device)

        generated_ids = model.generate(pixel_values, max_new_tokens=128, num_beams=4)
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if postprocess_german:
            texts = batch_correct(texts)

        return texts

    def _load_images(self, img_paths: List[str]) -> List:
        """Load images as PIL for processor."""
        from PIL import Image as PILImage
        from utils.image_utils import load_image
        images = []
        for path in img_paths:
            p = Path(path)
            if not p.is_absolute():
                p = Path('data') / p
            try:
                img = load_image(p, mode='rgb')
                images.append(PILImage.fromarray(img))
            except Exception:
                images.append(PILImage.new('RGB', (384, 64), color=255))
        return images

    def _meta_evaluate(self, tasks: List[Dict]) -> float:
        """Evaluate meta-learning by measuring CER after adaptation on each task."""
        from utils.metrics import compute_cer
        all_cers = []

        for task in tasks:
            learner = self.meta_model.clone()
            support = task['support']
            query = task['query']

            # Adapt
            support_images = self._load_images([s['image'] for s in support])
            support_texts = [s['text'] for s in support]
            for _ in range(self.inner_steps):
                loss = self._compute_loss(learner, support_images, support_texts)
                learner.adapt(loss)

            # Evaluate on query
            query_images = self._load_images([q['image'] for q in query])
            query_texts_ref = [q['text'] for q in query]

            from PIL import Image as PILImage
            import torch
            pixel_values = self.processor(
                images=query_images, return_tensors='pt'
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = learner.generate(pixel_values, max_new_tokens=128)
            hyps = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for h, r in zip(hyps, query_texts_ref):
                all_cers.append(compute_cer(h, r))

        return float(np.mean(all_cers)) if all_cers else 1.0

    def _save_meta_checkpoint(self, epoch: int, val_cer: float, output_dir: Path):
        """Save meta-learning checkpoint."""
        ckpt = {
            'epoch': epoch,
            'meta_model_state': self.meta_model.state_dict(),
            'val_cer': val_cer,
        }
        torch.save(ckpt, output_dir / 'meta_checkpoint_best.pt')
        # Save processor
        self.processor.save_pretrained(str(output_dir / 'processor'))
        logger.info(f"Saved meta checkpoint -> {output_dir / 'meta_checkpoint_best.pt'}")


# ---------------------------------------------------------------------------
# Manual MAML fallback (if learn2learn not installed)
# ---------------------------------------------------------------------------

class ManualMAML(nn.Module):
    """
    Simplified first-order MAML (Reptile variant) as fallback.
    Used when learn2learn is not available.
    """

    def __init__(self, model: nn.Module, lr: float = 0.01, steps: int = 5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.steps = steps

    def clone(self) -> 'ManualMAML':
        import copy
        cloned = ManualMAML(copy.deepcopy(self.model), self.lr, self.steps)
        return cloned

    def adapt(self, loss: torch.Tensor) -> None:
        """First-order adaptation step."""
        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        for p, g in zip(self.model.parameters(), grads):
            if g is not None:
                p.data -= self.lr * g.data

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


if __name__ == '__main__':
    logger.info("Testing meta-learning setup...")

    # Test task creation
    manifest = Path('data/processed/german_text/german_text_train.json')
    if manifest.exists():
        tasks = create_writer_tasks(manifest, n_support=5, n_query=5)
        logger.info(f"Created {len(tasks)} writer tasks")
        if tasks:
            logger.info(f"Example task: writer={tasks[0]['writer_id']}, "
                        f"support={len(tasks[0]['support'])}, "
                        f"query={len(tasks[0]['query'])}")
    else:
        logger.warning("Training manifest not found. Run prepare_iam_german.py first.")
