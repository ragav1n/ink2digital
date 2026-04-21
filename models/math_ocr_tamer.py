"""
TAMER (Tree-Aware Transformer) math OCR integration.
Phase 2 (Week 6): Replace Pix2Tex baseline with TAMER for better math recognition.

TAMER uses a tree-aware transformer architecture for handwritten mathematical
expression recognition, achieving state-of-the-art on CROHME and HME100K.

Available checkpoints (already in TAMER/lightning_logs/):
    version_0: CROHME w/o fusion, ExpRate=61.1%
    version_1: HME100K w/o fusion, ExpRate=68.5%
    version_3: HME100K w/ fusion, ExpRate=69.5% (BEST - use this)

Usage:
    from models.math_ocr_tamer import TAMERMathOCR
    tamer = TAMERMathOCR()
    latex = tamer.recognize(image_crop)
    latexes = tamer.recognize_batch([crop1, crop2, ...])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from loguru import logger


# ---------------------------------------------------------------------------
# TAMER integration
# ---------------------------------------------------------------------------

class TAMERMathOCR:
    """
    Wrapper for TAMER (Tree-Aware Transformer for Math Expression Recognition).

    Provides a unified interface for math OCR, with graceful fallback to
    Pix2Tex if TAMER initialization fails.

    Args:
        checkpoint_dir: Directory containing TAMER checkpoint (lightning_logs/version_X).
        device: Compute device.
        use_beam_search: Enable beam search decoding (higher quality, slower).
        beam_size: Beam size for decoding.
        fallback_to_pix2tex: If True, falls back to Pix2Tex if TAMER fails.
    """

    # Best available checkpoint
    DEFAULT_CHECKPOINT_DIR = 'TAMER/lightning_logs/version_3'
    DEFAULT_CONFIG = 'TAMER/config/hme100k.yaml'

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        use_beam_search: bool = True,
        beam_size: int = 10,
        fallback_to_pix2tex: bool = True,
    ):
        self.device = device
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.fallback_to_pix2tex = fallback_to_pix2tex

        ckpt_dir = Path(checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR)
        config = Path(config_path or self.DEFAULT_CONFIG)

        self.model = None
        self.model_type = None
        self._load(ckpt_dir, config)

    def _load(self, checkpoint_dir: Path, config_path: Path) -> None:
        """Load TAMER or fall back to Pix2Tex."""
        # Try TAMER first
        try:
            self.model = self._load_tamer(checkpoint_dir, config_path)
            self.model_type = 'tamer'
            logger.info("TAMER math OCR loaded successfully")
            return
        except Exception as e:
            logger.warning(f"TAMER load failed: {e}")

        # Fall back to Pix2Tex
        if self.fallback_to_pix2tex:
            try:
                self.model = self._load_pix2tex()
                self.model_type = 'pix2tex'
                logger.info("Fallback to Pix2Tex math OCR")
            except Exception as e:
                logger.error(f"Pix2Tex fallback also failed: {e}")
                self.model = None
                self.model_type = None

    def _load_tamer(self, checkpoint_dir: Path, config_path: Path):
        """Load TAMER from pretrained checkpoint."""
        tamer_root = Path('TAMER')
        if not tamer_root.exists():
            raise FileNotFoundError("TAMER directory not found")

        sys.path.insert(0, str(tamer_root.resolve()))

        ckpts = sorted(checkpoint_dir.glob('checkpoints/*.ckpt'))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files in {checkpoint_dir}/checkpoints/")
        ckpt_path = str(ckpts[-1])
        logger.info(f"Loading TAMER checkpoint: {ckpt_path}")

        import torch
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
        torch.serialization.add_safe_globals([ModelCheckpoint])
        from tamer.model.tamer import TAMER as TAMERModel
        from tamer.datamodule.vocab import CROHMEVocab

        # Checkpoint was saved from a wrapper (self.tamer_model = TAMER(...))
        # so all keys have a 'tamer_model.' prefix — strip it before loading.
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hparams = ckpt.get('hyper_parameters', {})
        state_dict = {
            k[len('tamer_model.'):]: v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('tamer_model.')
        }

        init_keys = {'d_model', 'growth_rate', 'num_layers', 'nhead',
                     'num_decoder_layers', 'dim_feedforward', 'dropout',
                     'dc', 'cross_coverage', 'self_coverage', 'vocab_size'}
        model = TAMERModel(**{k: v for k, v in hparams.items() if k in init_keys})
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"TAMER: {len(missing)} missing keys")
        if unexpected:
            logger.warning(f"TAMER: {len(unexpected)} unexpected keys")
        logger.info("TAMER weights loaded successfully")

        model.to(self.device)
        model.eval()

        # Store inference params from hparams for use in beam_search
        self._tamer_beam_size = hparams.get('beam_size', self.beam_size)
        self._tamer_max_len = hparams.get('max_len', 200)
        self._tamer_alpha = hparams.get('alpha', 1.0)
        self._tamer_early_stopping = hparams.get('early_stopping', False)
        self._tamer_temperature = hparams.get('temperature', 1.0)

        # Initialize vocabulary — version_1 dict has 245 words + 3 special = 248 = vocab_size
        dict_path = tamer_root / 'lightning_logs' / 'version_1' / 'dictionary.txt'
        if not dict_path.exists():
            raise FileNotFoundError(f"TAMER dictionary not found: {dict_path}")
        self._vocab = CROHMEVocab()
        self._vocab.init(str(dict_path))
        # Also initialize the module-level singleton used internally by generation_utils.
        # Must access via sys.modules because tamer.datamodule.__init__ shadows the name.
        import importlib
        _vocab_mod = importlib.import_module('tamer.datamodule.vocab')
        _vocab_mod.vocab.init(str(dict_path))
        logger.info(f"TAMER vocab loaded: {len(self._vocab)} tokens")

        return model

    def _load_pix2tex(self):
        """Load Pix2Tex as fallback."""
        from pix2tex.cli import LatexOCR
        return LatexOCR()

    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize math expression in a single image crop.

        Args:
            image: RGB numpy array (H, W, 3).

        Returns:
            LaTeX string for the math expression.
        """
        if self.model is None:
            return ''

        from utils.image_utils import numpy_to_pil
        pil_img = numpy_to_pil(image)

        if self.model_type == 'tamer':
            return self._tamer_inference(image)
        elif self.model_type == 'pix2tex':
            try:
                return self.model(pil_img) or ''
            except Exception as e:
                logger.debug(f"Pix2Tex error: {e}")
                return ''
        return ''

    def _tamer_inference(self, image: np.ndarray) -> str:
        """Run TAMER inference on an image."""
        import torch
        from torchvision import transforms
        from PIL import Image as PILImage

        # TAMER expects grayscale [b, 1, h, w] with specific normalization
        pil_img = PILImage.fromarray(image).convert('L')

        transform = transforms.Compose([
            transforms.Resize((128, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7931], std=[0.1738]),
        ])

        tensor = transform(pil_img).unsqueeze(0)  # [1, 1, 128, 512]
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)

        # img_mask: False = valid pixel, all False means no masking
        img_mask = torch.zeros(
            tensor.shape[0], tensor.shape[2], tensor.shape[3],
            dtype=torch.bool, device=device,
        )

        with torch.no_grad():
            try:
                hypotheses = self.model.beam_search(
                    img=tensor,
                    img_mask=img_mask,
                    beam_size=self._tamer_beam_size,
                    max_len=self._tamer_max_len,
                    alpha=self._tamer_alpha,
                    early_stopping=self._tamer_early_stopping,
                    temperature=self._tamer_temperature,
                )
                if not hypotheses:
                    return ''
                best = hypotheses[0]
                return self._decode_hypothesis(best.seq)
            except Exception as e:
                logger.debug(f"TAMER inference error: {e}")
                return ''

    def _decode_hypothesis(self, seq: list) -> str:
        """Decode a list of token indices to a LaTeX string."""
        EOS = self._vocab.EOS_IDX
        SOS = self._vocab.SOS_IDX
        PAD = self._vocab.PAD_IDX
        words = []
        for idx in seq:
            if idx in (EOS, SOS, PAD):
                break
            try:
                words.append(self._vocab.idx2word[idx])
            except KeyError:
                pass
        return ' '.join(words)

    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """Recognize math in a batch of images."""
        return [self.recognize(img) for img in images]

    def warm_up(self) -> None:
        """Warm up the model with a dummy inference."""
        dummy = np.ones((64, 256, 3), dtype=np.uint8) * 200
        self.recognize(dummy)
        logger.info("Math OCR model warmed up")


# ---------------------------------------------------------------------------
# TAMER fine-tuning for German math (Phase 2)
# ---------------------------------------------------------------------------

def prepare_tamer_training_data(
    crohme_dir: Path,
    output_dir: Path,
) -> None:
    """
    Prepare TAMER training data structure from CROHME.
    TAMER expects data in data/crohme/ format.

    This function converts our prepared math manifests to TAMER's expected format.
    """
    import json
    import shutil

    tamer_data_dir = Path('TAMER/data/crohme')
    tamer_data_dir.mkdir(parents=True, exist_ok=True)

    math_manifest = output_dir / 'math_train.json'
    if not math_manifest.exists():
        logger.error(f"Math manifest not found: {math_manifest}")
        logger.error("Run: python scripts/prepare_crohme.py first")
        return

    with open(math_manifest) as f:
        data = json.load(f)

    # Convert to TAMER's expected structure
    # TAMER needs: data/crohme/train_images/, train_labels.txt
    train_img_dir = tamer_data_dir / 'train_images'
    train_img_dir.mkdir(exist_ok=True)

    labels = []
    for sample in data.get('samples', []):
        img_path = Path(sample.get('image', ''))
        latex = sample.get('latex', '')
        if not img_path.exists():
            continue
        dest = train_img_dir / img_path.name
        shutil.copy2(str(img_path), str(dest))
        labels.append(f"{img_path.name}\t{latex}")

    with open(tamer_data_dir / 'train_labels.txt', 'w') as f:
        f.write('\n'.join(labels))

    logger.info(f"Prepared {len(labels)} samples for TAMER training -> {tamer_data_dir}")


if __name__ == '__main__':
    logger.info("Testing TAMER math OCR...")
    tamer = TAMERMathOCR()
    if tamer.model is not None:
        logger.info(f"Model type: {tamer.model_type}")
        test_img = np.ones((64, 256, 3), dtype=np.uint8) * 220
        result = tamer.recognize(test_img)
        logger.info(f"Test recognition result: '{result}'")
        tamer.warm_up()
    else:
        logger.error("No math OCR model could be loaded.")
        logger.info("Install pix2tex (pip install pix2tex) or ensure TAMER is properly set up.")
