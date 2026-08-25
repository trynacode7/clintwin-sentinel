"""
utils.py — Utility Functions for ICU Digital Twin System
========================================================
Logging, seeding, and device detection.

Phase B removal note: this module used to also define
generate_synthetic_data(), a second, independent synthetic-data
generator. It was confirmed dead code before removal — grepped for any
import/call site across the repo (`from utils import ...`, `import
utils`) and found none; the actual synthetic dataset generator this
project uses is backend/synthetic_data/generator.py, invoked via the
top-level generate_synthetic_data.py CLI script (a different module
with the same name coincidence). Removed rather than left in place,
per the Phase B instruction to eliminate undocumented duplicate
generation paths — there is now exactly one synthetic-data generator
in this repository.

Dependency note: `torch` is imported lazily inside the functions that
actually need it (set_seed, get_device, compute_class_weights) rather
than at module level. This is a deliberate, minimal change made during
the local-machine reconciliation phase: several modules (e.g.
preprocessing.py) only need setup_logging() from this file and have no
real dependency on torch, but an unconditional module-level `import
torch` forced torch to be installed just to import them at all. This
does not change behavior for any code path that does use torch — it
only changes *when* the import happens.
"""

import os
import random
import logging
import numpy as np


def setup_logging(level=logging.INFO):
    """Configure structured logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("icu_digital_twin")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-detect best available device (CUDA > CPU). Returns a torch.device."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.getLogger("icu_digital_twin").info(
            f"Using CUDA: {torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        logging.getLogger("icu_digital_twin").info("Using CPU")
    return device


def compute_class_weights(labels: np.ndarray):
    """
    Compute class weights for imbalanced binary classification.

    Uses inverse frequency weighting:
        weight_class_i = total_samples / (n_classes * count_class_i)

    Args:
        labels: Binary label array

    Returns:
        torch.Tensor with [weight_class_0, weight_class_1]
    """
    import torch

    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])

    weight_neg = n_total / (2.0 * n_neg)
    weight_pos = n_total / (2.0 * n_pos)

    return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
