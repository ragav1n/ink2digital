"""Device resolution helper.

Picks the best available torch device. Used so the pipeline runs unchanged on
the CUDA server *and* on an Apple Silicon Mac (MPS backend).

Invariant: on the CUDA server `torch.cuda.is_available()` is True, so a request
of ``'cuda'`` or ``'auto'`` resolves to ``'cuda'`` — server behaviour unchanged.
"""
from __future__ import annotations


def get_device(requested: str | None = 'auto') -> str:
    """Resolve a torch device string.

    - ``'auto'`` / ``None`` -> first available of cuda, mps, cpu.
    - ``'cuda'`` -> 'cuda' if available, else mps if available, else cpu.
    - ``'mps'``  -> 'mps' if available, else cpu.
    - ``'cpu'``  -> 'cpu'.
    """
    import torch

    def _cuda() -> bool:
        return torch.cuda.is_available()

    def _mps() -> bool:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    req = (requested or 'auto').lower()
    if req in ('auto', 'cuda'):
        if _cuda():
            return 'cuda'
        if _mps():
            return 'mps'
        return 'cpu'
    if req == 'mps':
        return 'mps' if _mps() else 'cpu'
    return req
