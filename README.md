# firefly — wildfire damage detection through a frozen fly brain

Can a *Drosophila melanogaster* connectome — never modified, never trained —
act as the feature extractor of a binary image classifier? A satellite tile's
99-dim retinal sample (54 brightness + 45 red-dominance patches) drives the
99 sensory neurons of the MaleCNS v1.0 circuit (166,700 neurons, 25.6M frozen
synapses, LIF dynamics); the 48-tick response is read as log1p spike counts of
a fixed set of 1,024 downstream neurons; a tiny 64-unit MLP decoder — the only
trained component — maps spikes to **undamaged vs damaged**.

Task framing: a plain ML benchmark on the Etkin satellite structure-damage
tiles (~18.7k unique tiles, binary collapse of 5 assessor damage grades,
chance = majority class). Numbers below are the committed benchmark; the
protocol (split, downstream pick, decoder family, seeds, ablations) was
committed before results.

## Results

| feature extractor | bal acc ↑ | AUC ↑ | acc |
|---|---|---|---|
| chance | 0.500 | 0.500 | majority |
| **frozen fly circuit** (the experiment) | _run pending_ | | |
| shuffled connectome (rewired control) | _run pending_ | | |
| all-excitatory (inhibition removed) | _run pending_ | | |
| random sparse wiring | _run pending_ | | |
| pooled pixels, no circuit | _run pending_ | | |

Lesion ablations (decoder frozen): optic lobe / central brain / VNC silenced,
a matched-size random neuron set, and no-image control — filled in
`results/benchmark.json` by the same run.

**Live demo** (k3s NodePort, LAN): `http://192.168.2.36:30181` — tile in,
damage score + connectome wave out, lesion switches replay the ablation live.

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
