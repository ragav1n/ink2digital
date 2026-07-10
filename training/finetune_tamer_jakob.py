"""
Fine-tune TAMER on Jakob's handwritten math.

Starts from TAMER/lightning_logs/version_3 (HME100K-pretrained, ExpRate 69.5%)
and fine-tunes on the 17/4 train/val split prepared at data/jakob_math_tamer/.

Output: checkpoint/tamer_jakob/  (best ckpt by val_ExpRate)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

# Add TAMER to import path (lit_tamer + datamodule)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'TAMER'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/jakob_math_tamer'))
    p.add_argument('--source-ckpt', type=Path,
                   default=Path('TAMER/lightning_logs/version_3/checkpoints/'
                                'epoch=55-step=175503-val_ExpRate=0.6954.ckpt'))
    p.add_argument('--output-dir', type=Path, default=Path('checkpoint/tamer_jakob'))
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--eval-batch-size', type=int, default=2)
    p.add_argument('--max-pixels', type=int, default=2_500_000,
                   help='Max image area in pixels (default 2.5M) — Jakob crops are wide')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    # Imports must follow sys.path mutation above
    from tamer.lit_tamer import LitTAMER
    from tamer.datamodule import HMEDatamodule

    pl.seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load source checkpoint ----------------------------------------------
    logger.info(f"Loading source checkpoint: {args.source_ckpt}")
    # PL 2.x can load PL 1.x checkpoints for state_dict + hparams, but some metadata
    # may not match — strict=False is safe here.
    try:
        model = LitTAMER.load_from_checkpoint(str(args.source_ckpt), strict=False)
    except Exception as e:
        logger.warning(f"load_from_checkpoint failed ({e}); reloading state_dict manually.")
        ckpt = torch.load(str(args.source_ckpt), map_location='cpu', weights_only=False)
        hparams = dict(ckpt.get('hyper_parameters', {}))
        model = LitTAMER(**hparams)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info(f"  missing={len(missing)} unexpected={len(unexpected)}")

    # Override for fine-tuning ---------------------------------------------------
    model.hparams.learning_rate = args.lr
    model.hparams.milestones = [int(args.epochs * 0.5), int(args.epochs * 0.8)]
    model.hparams.patience = max(4, args.epochs // 8)
    logger.info(f"Fine-tune hparams: lr={args.lr}, milestones={model.hparams.milestones}, "
                f"vocab_size={model.hparams.vocab_size}")

    # ---- Datamodule (uses test_folder for both val + test) -------------------
    dm = HMEDatamodule(
        folder=str(args.data.resolve()),
        test_folder='val',
        max_size=args.max_pixels,
        scale_to_limit=False,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
        scale_aug=False,
    )

    # ---- Callbacks -----------------------------------------------------------
    # val_ExpRate (exact-match) sticks at 0 with small data and long expressions,
    # so we select on val_loss instead. ExpRate is still logged for visibility.
    ckpt_cb = ModelCheckpoint(
        dirpath=str(args.output_dir / 'checkpoints'),
        filename='best-{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        default_root_dir=str(args.output_dir),
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        check_val_every_n_epoch=2,
        precision='32-true',
        callbacks=[ckpt_cb, lr_cb],
        gradient_clip_val=0.0,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)
    logger.info(f"Best val_ExpRate: {ckpt_cb.best_model_score} -> {ckpt_cb.best_model_path}")

    # Promote best checkpoint to a stable path the pipeline can use
    if ckpt_cb.best_model_path:
        target_dir = args.output_dir / 'lightning_logs' / 'jakob' / 'checkpoints'
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / Path(ckpt_cb.best_model_path).name
        shutil.copyfile(ckpt_cb.best_model_path, dest)
        # Copy hparams + the source config so TAMERMathOCR can locate them
        (args.output_dir / 'lightning_logs' / 'jakob' / 'hparams.yaml').write_text(
            (ROOT / 'TAMER' / 'lightning_logs' / 'version_3' / 'hparams.yaml').read_text()
        )
        logger.info(f"Promoted best ckpt -> {dest}")


if __name__ == '__main__':
    main()
