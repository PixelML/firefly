"""Firefly experiment: binary wildfire-damage classification through a frozen
Drosophila connectome, benchmarked like any ML model.

Conditions (same split, same frozen 1,024-downstream pick, same decoder family):
  circuit              the real MaleCNS v1.0 wiring (the experiment)
  shuffled_connectome  per-row degree-preserving rewiring of the real weights
  all_excitatory       |W|: GABA inhibition removed
  random_sparse        same nnz, random positions, weights from |W| distribution
  pixel_mlp            MLP on the 99-dim retinal sample, circuit skipped
  label_shuffle        real features, shuffled TRAIN labels, eval on true val labels
Lesion ablations (frozen circuit decoder): silence optic lobe / central brain /
VNC / a matched-size random neuron set / no sensory input at all.

Primary metrics: balanced accuracy and ROC-AUC (chance 0.5), with bootstrap
CIs. Everything seeded; split, downstream pick and protocol are committed in
this file before results are generated (pre-registration by commit order).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .connectome import load_graph
from .retina import build_drive, pick_downstream, retinal_sample, run_features
from .sim import FlySim

STEPS = 48
N_DOWNSTREAM = 1024
DECODER_SEEDS = (0, 1, 2)
EPOCHS = 30
BATCH = 256
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 7
SPLIT_SEED = 0
PICK_DRIVES = 512
GALLERY_PER_BUCKET = 6
CLASSES = ["undamaged", "damaged"]


# ---------------------------------------------------------------- data

def load_tiles(shard_glob: str):
    """All tiles: retinal features, binary label, dedupe by content hash."""
    feats_l, y5_l, locs, seen = [], [], [], set()
    t0 = time.time()
    n_raw = 0
    for shard in sorted(glob.glob(shard_glob)):
        d = pq.read_table(shard, columns=["image", "label"]).to_pandas()
        imgs = d["image"]
        for i in range(len(d)):
            n_raw += 1
            raw = imgs[i]["bytes"] if isinstance(imgs[i], dict) else imgs[i]
            h = hashlib.md5(raw).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            feats_l.append(retinal_sample(Image.open(io.BytesIO(raw)).convert("RGB")))
            y5_l.append(int(d["label"][i]))
            locs.append((shard, i))
        print(json.dumps({"evt": "shard", "file": Path(shard).name,
                          "unique": len(feats_l)}), flush=True)
    X = torch.as_tensor(np.stack(feats_l))
    y5 = np.array(y5_l)
    y = (y5 > 0).astype(np.int64)   # binary: any damage vs none
    print(json.dumps({"evt": "data", "raw": n_raw, "unique": len(y),
                      "undamaged": int((y == 0).sum()), "damaged": int((y == 1).sum()),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    return X, y, y5, locs


def stratified_split(y: np.ndarray, seed: int, frac: float = 0.85):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(idx.size * frac))
        tr.append(idx[:k]); va.append(idx[k:])
    tr = np.concatenate(tr); va = np.concatenate(va)
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


# ---------------------------------------------------------------- decoder + metrics

def mlp_train(Xtr, ytr, seed, dims):
    torch.manual_seed(seed)
    dev = Xtr.device
    dec = torch.nn.Sequential(torch.nn.Linear(dims, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 2)).to(dev)
    w = torch.tensor([int((ytr == c).sum()) for c in (0, 1)], dtype=torch.float32, device=dev)
    w = w.sum() / w.clamp_min(1)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(EPOCHS):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(dec(Xtr[idx]), ytr[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return dec


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with tie handling (no sklearn dependency)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(y_true, scores, threshold=0.5):
    pred = (scores >= threshold).astype(np.int64)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else float("nan")
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else float("nan")
        f = 2 * p * r / (p + r) if p + r else float("nan")
        return [round(p, 4), round(r, 4), round(f, 4)]
    bacc = ((tp / (tp + fn)) if tp + fn else 0.0) + ((tn / (tn + fp)) if tn + fp else 0.0)
    return {
        "accuracy": round(float((pred == y_true).mean()), 4),
        "balanced_accuracy": round(bacc / 2, 4),
        "roc_auc": round(roc_auc(y_true, scores), 4),
        "confusion_undamaged_damaged": [[tn, fp], [fn, tp]],
        "precision_recall_f1_per_class": {"undamaged": prf(tn, fp, fn),
                                          "damaged": prf(tp, fn, tn)},
    }


def bootstrap_ci(y_true, scores, n_boot=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ("accuracy", "balanced_accuracy", "roc_auc")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = evaluate(y_true[idx], scores[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)] for k, v in samples.items()}


def score_out(dec, X, mu, sd):
    with torch.no_grad():
        xn = (X - mu) / sd
        return torch.softmax(dec(xn), dim=-1)[:, 1].cpu().numpy()


def fit_decoder(Xtr, ytr, Xva, yva, seed, tag):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    dec = mlp_train((Xtr - mu) / sd, ytr, seed, Xtr.shape[1])
    s = score_out(dec, Xva, mu, sd)
    m = evaluate(yva, s)
    print(json.dumps({"evt": "result", "cond": tag, "seed": seed, **m}), flush=True)
    return dec, (mu, sd), s, m


# ---------------------------------------------------------------- circuit variants

def variant_w(graph, kind, seed=0):
    """Rewiring controls: same neurons, same scale, different (or no) biology."""
    z = graph["arrays"]
    indptr = z["indptr"].astype(np.int64)
    indices = z["indices"].astype(np.int64).copy()
    values = z["values"].astype(np.float32)
    if kind == "real":
        ind, val = indices, values
    elif kind == "shuffled":          # per-row degree-preserving rewiring
        rng = np.random.default_rng(seed)
        for r in range(len(indptr) - 1):
            seg = indices[indptr[r]:indptr[r + 1]]
            rng.shuffle(seg)
        ind, val = indices, values
    elif kind == "all_excitatory":    # inhibition removed
        ind, val = indices, np.abs(values)
    elif kind == "random_sparse":     # same nnz, random positions, |W| weights
        rng = np.random.default_rng(seed)
        n = len(indptr) - 1
        nnz = len(values)
        rows = np.sort(rng.integers(0, n, nnz))
        cols = rng.integers(0, n, nnz)
        indptr2 = np.searchsorted(rows, np.arange(n + 1))
        ind, val = cols, np.abs(rng.choice(values, nnz))
        return indptr2.astype(np.int64), ind.astype(np.int64), val.astype(np.float32)
    else:
        raise ValueError(kind)
    return indptr, ind, val


def swap_w(sim, w3):
    indptr, ind, val = w3
    sim.W = torch.sparse_csr_tensor(
        torch.from_numpy(indptr.astype(np.int32)),
        torch.from_numpy(ind.astype(np.int32)),
        torch.from_numpy(val.astype(np.float32)),
        size=(sim.N, sim.N), dtype=sim.dtype, device=sim.device)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default="/library/datasets/fire_sat/etkin/*.parquet")
    ap.add_argument("--out", type=Path, default=Path("/library/datasets/firefly"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "samples").mkdir(parents=True, exist_ok=True)

    Xpix, y, y5, locs = load_tiles(args.data_glob)
    tr, va = stratified_split(y, SPLIT_SEED)
    meta = {
        "task": "binary wildfire-damage classification of satellite tiles "
                "(undamaged vs damaged) through a frozen Drosophila connectome",
        "circuit": "MaleCNS v1.0 frozen; 48-tick response; log1p spike counts of a "
                   f"fixed {N_DOWNSTREAM}-downstream-neuron set picked once from the "
                   f"busiest non-sensory neurons on the first {PICK_DRIVES} training drives",
        "decoder": "standardized 64-unit MLP, 2 classes, class-weighted CE, 30 epochs",
        "split": f"stratified 85/15 tile-level, seed {SPLIT_SEED}; tiles hash-deduped; "
                 "no scene metadata in source, so near-duplicate scene leakage is possible",
        "threshold": 0.5, "scores_calibrated": False,
        "decoder_seeds": list(DECODER_SEEDS), "bootstrap_resamples": BOOTSTRAP,
        "primary_metrics": "balanced_accuracy, roc_auc (chance 0.5)",
        "dataset": "Etkin et al. structure-damage tiles via "
                   "kevincluo/structure_wildfire_damage_classification (apache-2.0)",
        "recipe_after": "jerryjliu/fly_ocr (MIT), nftechie/doomfly (MIT)",
        "data": {"unique_tiles": int(len(y)), "undamaged": int((y == 0).sum()),
                 "damaged": int((y == 1).sum()), "train": int(len(tr)), "val": int(len(va))},
    }

    sim = FlySim(load_graph(), dev)
    print(json.dumps({"evt": "circuit", "device": dev, "neurons": sim.N}), flush=True)

    ftr, fva = Xpix[tr].to(dev), Xpix[va].to(dev)
    ytr_t = torch.as_tensor(y[tr], device=dev)
    yva = y[va]

    # frozen downstream pick: train drives only, never labels
    t1 = time.time()
    downstream = pick_downstream(sim, build_drive(sim, ftr[:PICK_DRIVES].cpu().numpy()),
                                 STEPS, N_DOWNSTREAM)
    print(json.dumps({"evt": "downstream_pick", "seconds": round(time.time() - t1, 1)}), flush=True)

    # circuit + rewiring controls, same downstream pick, 3 decoder seeds each
    conditions = {}
    real = {}
    for kind in ("real", "shuffled", "all_excitatory", "random_sparse"):
        swap_w(sim, variant_w(load_graph(), kind, seed=0))
        t2 = time.time()
        Xtr = run_features(sim, build_drive(sim, ftr.cpu().numpy()), STEPS, downstream)
        Xva = run_features(sim, build_drive(sim, fva.cpu().numpy()), STEPS, downstream)
        accs, s0 = [], None
        for seed in DECODER_SEEDS:
            dec, (mu, sd), s, m = fit_decoder(Xtr, ytr_t, Xva, yva, seed, kind)
            accs.append(m["accuracy"])
            if seed == 0:
                s0 = s
                if kind == "real":
                    real = {"dec": dec, "mu": mu, "sd": sd, "scores": s}
        conditions[kind] = {**m, "accuracy_seeds": [round(a, 4) for a in accs],
                            "ci": bootstrap_ci(yva, s0)}
        print(json.dumps({"evt": "condition_done", "cond": kind,
                          "seconds": round(time.time() - t2, 1)}), flush=True)

    # pixel baseline (no circuit): same split, same decoder family, 3 seeds
    accs, s0 = [], None
    for seed in DECODER_SEEDS:
        _, _, s, m = fit_decoder(ftr, ytr_t, fva, yva, seed, "pixel_mlp")
        accs.append(m["accuracy"])
        if seed == 0:
            s0 = s
    conditions["pixel_mlp"] = {**m, "accuracy_seeds": [round(a, 4) for a in accs],
                               "ci": bootstrap_ci(yva, s0)}

    # label-shuffle control: real features, shuffled train labels, true val labels
    rng = np.random.default_rng(0)
    ytr_shuf = torch.as_tensor(rng.permutation(y[tr]), device=dev)
    _, _, s_shuf, m_shuf = fit_decoder(ftr, ytr_shuf, fva, yva, 0, "label_shuffle")
    conditions["label_shuffle"] = {"roc_auc": m_shuf["roc_auc"],
                                   "note": "shuffled train labels, true val labels; "
                                           "AUC ~0.5 means no leakage"}

    # lesion ablations: frozen real-circuit decoder, val features under silence
    def lesion_metrics(tag):
        Xva_l = run_features(sim, build_drive(sim, fva.cpu().numpy()), STEPS, downstream)
        s = score_out(real["dec"], Xva_l, real["mu"], real["sd"])
        m = evaluate(yva, s)
        m["delta_balanced_accuracy"] = round(
            m["balanced_accuracy"] - conditions["real"]["balanced_accuracy"], 4)
        conditions.setdefault("lesions", {})[tag] = m
        print(json.dumps({"evt": "lesion", "cond": tag,
                          "balanced_accuracy": m["balanced_accuracy"],
                          "delta": m["delta_balanced_accuracy"]}), flush=True)

    sim.set_silence(0); lesion_metrics("optic_lesioned")
    sim.set_silence(1); lesion_metrics("central_lesioned")
    sim.set_silence(2); lesion_metrics("vnc_lesioned")
    n_optic = int((sim.region == 0).sum())
    rng = np.random.default_rng(1)
    sim.set_silence_neurons(torch.as_tensor(
        rng.choice(sim.N, n_optic, replace=False), device=dev))
    lesion_metrics("random_matched_optic")
    sim.set_silence(None)
    Xva_zero = run_features(sim, torch.zeros(sim.N, fva.shape[0], device=dev, dtype=sim.dtype),
                            STEPS, downstream)
    s_zero = score_out(real["dec"], Xva_zero, real["mu"], real["sd"])
    conditions.setdefault("lesions", {})["no_sensory_input"] = {
        **evaluate(yva, s_zero),
        "note": "drive replaced with zeros: what the decoder reads without an image"}

    # held-out gallery: hits and misses of the serving decoder
    gallery = {"undamaged": {"hit": [], "miss": []}, "damaged": {"hit": [], "miss": []}}
    for j, gi in enumerate(va):
        pred = int(real["scores"][j] >= 0.5)
        bucket = gallery[CLASSES[int(y[gi])]]["hit" if pred == y[gi] else "miss"]
        if len(bucket) < GALLERY_PER_BUCKET:
            bucket.append({"global_row": int(gi), "y5": int(y5[gi]),
                           "score": round(float(real["scores"][j]), 4)})
    save_gallery(args.out, locs, gallery)

    (args.out / "results" / "benchmark.json").write_text(
        json.dumps({"meta": meta, "conditions": conditions}, indent=2) + "\n")
    np.savez(args.out / "readout.npz",
             W1=real["dec"][0].weight.detach().cpu().numpy(),
             b1=real["dec"][0].bias.detach().cpu().numpy(),
             W2=real["dec"][2].weight.detach().cpu().numpy(),
             b2=real["dec"][2].bias.detach().cpu().numpy(),
             mu=real["mu"].cpu().numpy(), sd=real["sd"].cpu().numpy(),
             downstream=downstream.cpu().numpy().astype(np.int64))
    print(json.dumps({"evt": "done",
                      "circuit_balanced_acc": conditions["real"]["balanced_accuracy"],
                      "circuit_auc": conditions["real"]["roc_auc"],
                      "pixel_balanced_acc": conditions["pixel_mlp"]["balanced_accuracy"],
                      "shuffled_auc": conditions["shuffled"]["roc_auc"],
                      "optic_lesion_delta":
                          conditions["lesions"]["optic_lesioned"]["delta_balanced_accuracy"],
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def save_gallery(out: Path, locs, gallery):
    """Materialize the held-out gallery tiles as JPEGs + manifest."""
    gdir = out / "samples"
    want = {}
    for cls_name, buckets in gallery.items():
        for kind, items in buckets.items():
            for it in items:
                want[it["global_row"]] = (cls_name, kind, it)
    by_shard = {}
    for gr, (shard, row) in enumerate(locs):
        if gr in want:
            by_shard.setdefault(shard, []).append((row, gr))
    manifest = []
    for shard, pairs in by_shard.items():
        d = pq.read_table(shard, columns=["image"]).to_pandas()
        imgs = d["image"]
        for row, gr in pairs:
            raw = imgs[row]["bytes"] if isinstance(imgs[row], dict) else imgs[row]
            cls_name, kind, it = want[gr]
            fname = f"{cls_name}_{kind}_{gr}.jpg"
            Image.open(io.BytesIO(raw)).convert("RGB").save(gdir / fname, quality=88)
            manifest.append({"id": f"v{gr}", "file": fname,
                             "true_label": 0 if cls_name == "undamaged" else 1,
                             "y5": it["y5"], "kind": kind,
                             "fly_score": it["score"]})
    (gdir / "manifest.json").write_text(json.dumps(
        {"classes": CLASSES, "tiles": manifest}, indent=1) + "\n")
    print(json.dumps({"evt": "gallery", "tiles": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
