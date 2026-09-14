"""Batched LIF simulation over the fixed MaleCNS v1.0 wiring.

Approximate neural dynamics (engineering choice, doomfly-style): synapses are
signed contact-count strengths (GABA edges negative, from the dataset's
consensus neurotransmitter predictions); state is a leaky integrate-and-fire
membrane per neuron under per-neuron homeostatic threshold control. An image's
99-dim retinal sample drives the 99 sensory neurons directly (drive, not
spikes). Per-tick spike counts are tracked per brain region (optic / central /
VNC) and for arbitrary neuron subsets (feature sets, telemetry samples), and a
region can be silenced live (its spikes suppressed before propagation) for
lesion ablations. The wiring is never modified or trained.
"""
from __future__ import annotations

import numpy as np
import torch

# Retinal patch grid: 54 brightness patches -> R1-R6 rows, 45 red-dominance
# patches -> R8 rows (9 columns x 9 rows of patches total).
N_PATCH_R16 = 54   # 6 rows x 9 cols
N_PATCH_R8 = 45    # 5 rows x 9 cols


class FlySim:
    def __init__(self, graph: dict, device, decay: float = 0.85, gain: float = 0.15,
                 noise: float = 0.01, homeo_k: float = 12.0, target_rate: float = 0.03,
                 dtype=torch.float32):
        z = graph["arrays"]
        self.device = device
        self.dtype = dtype
        self.N = graph["n_nodes"]
        n = self.N
        crow = torch.from_numpy(z["indptr"].astype(np.int32))
        col = torch.from_numpy(z["indices"].astype(np.int32))
        val = torch.from_numpy(z["values"].astype(np.float32))
        self.W = torch.sparse_csr_tensor(
            crow, col, val, size=(n, n), dtype=dtype, device=device)
        theta = gain * torch.sqrt(torch.from_numpy(z["theta_scale"].astype(np.float32)) + 1.0)
        self.theta = theta.to(device)
        sidx = z["sensory_idx"]
        self.rows_r16 = torch.from_numpy(sidx[:N_PATCH_R16].astype(np.int64)).to(device)
        self.rows_r8 = torch.from_numpy(sidx[N_PATCH_R16:].astype(np.int64)).to(device)
        region = torch.from_numpy(z["region"].astype(np.int64)).to(device)
        self.region = region
        self._region_masks = [(region == c).to(dtype) for c in (0, 1, 2)]
        self._alive = torch.ones(n, 1, device=device, dtype=dtype)
        self.decay, self.noise = decay, noise
        self.homeo_k, self.target_rate = homeo_k, target_rate
        self.silenced = None   # region code to lesion (0 optic, 1 central, 2 VNC), or None

    def set_silence(self, region):
        """Silence a brain region live: its spikes are suppressed before propagation."""
        if region is None:
            self.silenced = None
            self._alive = 1.0
            return
        self.silenced = region
        self._alive = 1.0 - self._region_masks[region].unsqueeze(1)

    def set_silence_neurons(self, idx: torch.Tensor):
        """Silence an arbitrary neuron subset (matched-random lesion control)."""
        self.silenced = "custom"
        mask = torch.ones(self.N, 1, device=self.device, dtype=self.dtype)
        mask[idx.to(self.device)] = 0.0
        self._alive = mask

    def sensory_rows(self) -> torch.Tensor:
        return torch.cat([self.rows_r16, self.rows_r8])

    @torch.no_grad()
    def run(self, drive: torch.Tensor, steps: int,
            count_idx: torch.Tensor | None = None):
        """Simulate `steps` ticks for a [N, B] drive; returns decision signals."""
        batch = drive.shape[1]
        v = torch.rand(self.N, batch, device=self.device, dtype=self.dtype) * 0.05
        s_prev = torch.zeros(self.N, batch, device=self.device, dtype=self.dtype)
        region_wave = torch.zeros(3, steps, device=self.device, dtype=self.dtype)
        total_spikes = torch.zeros(steps, device=self.device)
        counts = (torch.zeros(count_idx.numel(), batch, device=self.device, dtype=self.dtype)
                  if count_idx is not None else None)
        noise = self.noise * torch.randn(self.N, 1, device=self.device, dtype=self.dtype)
        rate = torch.full((self.N, 1), self.target_rate, device=self.device, dtype=self.dtype)
        theta_base = self.theta.unsqueeze(1)
        alive = self._alive
        for t in range(steps):
            cur = torch.sparse.mm(self.W, s_prev)
            v = self.decay * v + cur + drive + noise
            s = (v > theta_base * torch.exp(self.homeo_k * (rate - self.target_rate))).to(self.dtype)
            if self.silenced is not None:
                s = s * alive
            v = v.masked_fill(s.bool(), 0.0)
            rate = 0.97 * rate + 0.03 * s.mean(1, keepdim=True)
            s_prev = s
            for ri, rmask in enumerate(self._region_masks):
                region_wave[ri, t] = (s * rmask.unsqueeze(1)).sum()
            if counts is not None:
                counts += s[count_idx]
            total_spikes[t] = s.sum()
        return {
            "total_spikes": total_spikes,
            "counts": counts,
            "region_wave": region_wave,
        }
