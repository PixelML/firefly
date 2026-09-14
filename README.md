# firefly — wildfire damage detection through a frozen fly brain

Can a *Drosophila melanogaster* connectome — never modified, never trained —
act as the feature extractor of a binary image classifier? A satellite tile's
99-dim retinal sample (54 brightness + 45 red-dominance patches) drives the
99 sensory neurons of the MaleCNS v1.0 circuit (166,700 neurons, 25.6M frozen
synapses, LIF dynamics); the 48-tick response is read as log1p spike counts of
a fixed set of 1,024 downstream neurons; a tiny 64-unit MLP decoder — the only
trained component — maps spikes to **undamaged vs damaged**.

Task framing: a plain ML benchmark on the Etkin satellite structure-damage
tiles (binary collapse of 5 assessor damage grades). Numbers below are the
committed benchmark; the protocol (split, downstream pick, decoder family,
seeds, ablations) was committed before any binary result was generated.

## Results

18,318 unique tiles (hash-deduped; 1,222 undamaged / 17,096 damaged — the
majority class predicts 93.3% accuracy, so raw accuracy is meaningless here;
balanced accuracy and ROC-AUC are the primary metrics). Same split, same
frozen 1,024-downstream pick, same decoder family, 3 decoder seeds,
1000-resample bootstrap CIs in `results/benchmark.json`.

| feature extractor | bal acc ↑ | AUC ↑ |
|---|---|---|
| chance | 0.500 | 0.500 |
| **frozen fly circuit** (the experiment) | **0.588** [0.541–0.614] | **0.622** |
| shuffled connectome (degree-preserving rewiring) | 0.585 | 0.625 |
| all-excitatory (inhibition removed) | 0.609 | 0.642 |
| random sparse wiring (same nnz) | 0.510 | 0.632 |
| pooled pixels, no circuit | **0.655** | **0.711** |
| label shuffle (leakage check) | — | 0.400 |

### Verdict

1. **The specific wiring carries no detectable task-relevant structure.**
   The real connectome (AUC 0.622) is indistinguishable from a degree-
   preserving rewiring (0.625), from removing all inhibition (0.642), and
   from a random sparse matrix with matched nnz (0.632). Whatever lifts the
   score above chance lives in the retinal encoding + generic nonlinear
   mixing dynamics, not in the connectome's connectivity.
2. **Pooled pixels beat the circuit** (AUC 0.711 vs 0.622; balanced acc
   0.655 vs 0.588) — the 48-tick fixed circuit is a lossy, noisy transform
   of its 99-dim drive.
3. **Lesions have no specificity.** Silencing the optic lobe, the central
   brain, the VNC, or a matched-size random neuron set all collapse the
   frozen decoder to the majority-class predictor (bal acc ≈ 0.50). The
   above-chance signal needs the whole circuit; it is not specifically
   visual. Removing the image entirely does the same.
4. **No leakage detected** (label-shuffle control AUC 0.400, i.e. the
   chance band; strong leakage would push AUC ≫ 0.5).

An interesting, decisive negative result: as driven here (99-patch retinal
sample → 99 sensory neurons), the fly brain is interchangeable with a random
fixed projection. What would falsify that conclusion: driving it with
anatomically motivated inputs (e.g. columnar retina geometry, motion) or
tasks closer to what the optic lobe computes.

**Live demo** (k3s NodePort, LAN): `http://192.168.2.36:30181` — tile in,
damage score + connectome wave out, lesion switches replay the ablation live,
held-out hits/misses gallery included.
## Method

```
tile --> retinal sample (99 patches) --> 99 sensory neurons (drive)
      --> frozen MaleCNS v1.0, 48 LIF ticks
      --> log1p spike counts, fixed 1,024 downstream neurons
      --> standardize --> 64-unit MLP (only trained part) --> P(damaged)
```

- Downstream pick: busiest non-sensory neurons by total spikes on the first
  512 training drives (drive-derived only, never labels), frozen for every
  condition including controls.
- Split: stratified 85/15, seed 0, hash-deduped tiles. The source has no scene
  metadata, so near-duplicate scene leakage across the split is possible — a
  stated limitation, identical for every condition, so comparisons stand.
- Metrics: balanced accuracy + ROC-AUC at threshold 0.5 (scores not
  calibrated), 1000-resample bootstrap CIs, 3 decoder seeds.
- Controls: shuffled (degree-preserving rewired) connectome, all-excitatory
  (inhibition removed), random sparse wiring, pooled-pixel MLP on the same
  retinal sample, label-shuffle leakage check.

## Run

```sh
python3 -m firefly.train --device cuda      # trains, benchmarks, writes results/
python3 -m uvicorn serve.app:app --port 8000  # FIREFLY_OUT=/library/datasets/firefly
kubectl apply -f deploy/namespace.yaml -f deploy/web.yaml   # NodePort 30181
```

Training reads `/library/datasets/fire_sat/etkin/*.parquet` and the processed
MaleCNS cache (`/library/datasets/malecns_v1/processed-firefly/`); serving
reads only `readout.npz`, `results/benchmark.json`, `samples/` from
`FIREFLY_OUT` — ~1 s per tile on CPU.

## What this is not

Not biologically validated vision: the retinal mapping is an engineering
choice, dynamics are doomfly-style approximations, and pooled pixels beat the
circuit (see table) — the question here is whether *connectome wiring carries
task-relevant structure*, not whether a fly brain is a good camera.

## Credits

MaleCNS v1.0: HHMI Janelia + Google (CC-BY) · data: Etkin et al. via
kevincluo/structure_wildfire_damage_classification (apache-2.0) · recipe after
jerryjliu/fly_ocr (MIT) and nftechie/doomfly (MIT). Companion playground with
the game-based demos: seanphan/flyt3.
