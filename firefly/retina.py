"""Retinal encoding and frozen-circuit feature extraction (fly_ocr recipe).

retinal_sample: 99-dim drive — 54 brightness patches (6x9 grid, R1-R6 rows)
and 45 red-dominance patches (5x9 grid, R8 rows). Pixels never reach the
decoder directly; they only drive the 99 sensory neurons.

build_drive / pick_downstream / run_features: the tile -> spike-count features
path. The downstream set is picked ONCE from the busiest non-sensory neurons
on the first 512 training drives (drive-derived only, never labels) and is
then frozen for every condition in the experiment.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .sim import N_PATCH_R16, N_PATCH_R8


def retinal_sample(img: Image.Image) -> np.ndarray:
    """99-dim: 54 brightness patches -> R1-R6 rows, 45 red-dominance -> R8 rows."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    def pool(ch, rows, cols):
        im = Image.fromarray((np.clip(ch, 0, 1) * 255).astype(np.uint8)).resize((cols * 8, rows * 8))
        b = np.asarray(im, dtype=np.float32) / 255.0
        bh, bw = b.shape[0] // rows, b.shape[1] // cols
        return b[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))

    bright = pool(np.asarray(img.convert("L"), dtype=np.float32) / 255.0, 6, 9)
    red = np.clip(a[:, :, 0] - (a[:, :, 1] + a[:, :, 2]) / 2, 0, 1)
    redd = pool(red, 5, 9)
    assert bright.size == N_PATCH_R16 and redd.size == N_PATCH_R8
    return np.concatenate([bright.reshape(-1), redd.reshape(-1)]).astype(np.float32)


def build_drive(sim, feats: np.ndarray) -> torch.Tensor:
    """[B, 99] retinal samples -> [N, B] sensory drive (scaled 2.4, as fly_ocr)."""
    B = feats.shape[0]
    d = torch.zeros(sim.N, B, device=sim.device, dtype=sim.dtype)
    srows = sim.sensory_rows()
    vals = torch.as_tensor(feats, dtype=sim.dtype, device=sim.device)
    d[srows] = vals.T * 2.4
    return d


@torch.no_grad()
def pick_downstream(sim, drive: torch.Tensor, steps: int, n: int = 1024):
    """The n busiest non-sensory neurons by total spike count on sample drives."""
    tot = torch.zeros(sim.N, device=sim.device)
    all_idx = torch.arange(sim.N, device=sim.device)
    for i in range(0, drive.shape[1], 128):
        out = sim.run(drive[:, i:i + 128], steps, count_idx=all_idx)
        tot += out["counts"].sum(1)
    tot[sim.sensory_rows()] = 0
    return torch.topk(tot, n).indices


@torch.no_grad()
def run_features(sim, drive: torch.Tensor, steps: int,
                 downstream: torch.Tensor, chunk: int = 256):
    """log1p spike counts of `downstream` neurons for every drive column."""
    counts = torch.zeros(downstream.numel(), drive.shape[1], device=sim.device)
    for i in range(0, drive.shape[1], chunk):
        out = sim.run(drive[:, i:i + chunk], steps, count_idx=downstream)
        counts[:, i:i + chunk] = out["counts"]
    return torch.log1p(counts.T)
