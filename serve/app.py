"""Firefly demo server: tile in -> the frozen fly brain's damage score out.

The experiment as a service. A satellite tile's 99-dim retinal sample drives
the frozen MaleCNS v1.0 circuit for 48 ticks; log1p spike counts of the fixed
1,024-downstream set go through the trained 64-unit MLP decoder; the page gets
P(damaged) plus the live per-region spike wave for the connectome view.
Lesion switches silence brain regions live, so the ablation can be replayed
in the browser. Endpoints are sync so FastAPI runs the sim in its threadpool.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from firefly.connectome import load_graph
from firefly.retina import build_drive, retinal_sample, run_features
from firefly.sim import FlySim

OUT = Path(os.environ.get("FIREFLY_OUT", "/out"))
STATIC_DIR = Path(os.environ.get("FIREFLY_STATIC", "/app/serve/static"))
STEPS = 48

app = FastAPI(title="firefly")
state: dict = {}


def _no_nan(o):
    """Strict JSON: NaN/Inf are not representable; they read as missing."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    return o


def build_brain_sample(meta: dict, n_optic=1300, n_central=1300, n_vnc=800) -> dict:
    """Stratified neuron sample with stylized 3D positions for the connectome view."""
    z = meta["arrays"]
    region = z["region"]
    rng = np.random.default_rng(42)
    picks = []
    for code, n in ((0, n_optic), (1, n_central), (2, n_vnc)):
        idxs = np.nonzero(region == code)[0]
        take = min(n, len(idxs))
        picks.extend(rng.choice(idxs, size=take, replace=False).tolist())
    idx = np.array(sorted(picks), dtype=np.int64)
    reg = region[idx]
    pts = np.zeros((len(idx), 3), dtype=np.float32)
    r = rng.random(len(idx))
    for code, mask in ((0, reg == 0), (1, reg == 1), (2, reg == 2)):
        k = int(mask.sum())
        if k == 0:
            continue
        g = lambda s, sc: rng.normal(0, sc, k).astype(np.float32)  # noqa: E731
        if code == 0:    # two optic lobes flanking the brain
            pts[mask, 0] = np.where(r[mask] < 0.5, -1.0, 1.0) + g(0, 0.28)
            pts[mask, 1] = 0.72 + g(0, 0.26)
            pts[mask, 2] = g(0, 0.26)
        elif code == 1:  # central brain sphere
            pts[mask, 0] = g(0, 0.42)
            pts[mask, 1] = 0.88 + g(0, 0.30)
            pts[mask, 2] = g(0, 0.42)
        else:            # ventral nerve cord below
            pts[mask, 0] = g(0, 0.16)
            pts[mask, 1] = -0.62 + g(0, 0.52)
            pts[mask, 2] = g(0, 0.16)
    return {"idx": idx, "points": pts, "region": reg}


@app.on_event("startup")
def startup():
    dev = os.environ.get("FIREFLY_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_graph()
    state["meta"] = meta
    state["sim"] = FlySim(meta, dev)
    state["dev"] = dev
    state["brain"] = build_brain_sample(meta)
    state["brain_idx_t"] = torch.from_numpy(state["brain"]["idx"]).to(dev)
    # multi-source BFS hop depth from all sensory inputs along real wiring
    n = meta["n_nodes"]
    z = meta["arrays"]
    reach = torch.zeros(n, device=dev)
    reach[torch.from_numpy(z["sensory_idx"].astype(np.int64)).to(dev)] = 1.0
    binW = torch.sparse_csr_tensor(
        torch.from_numpy(z["indptr"].astype(np.int32)),
        torch.from_numpy(z["indices"].astype(np.int32)),
        torch.ones(len(z["indices"]), dtype=torch.float32),
        size=(n, n), device=dev)
    depth = torch.full((n,), 1e9, device=dev)
    depth[reach.bool()] = 0
    frontier = reach
    for level in range(1, 65):
        nxt = ((torch.sparse.mm(binW, frontier.unsqueeze(1)).squeeze(1) > 0) & (depth > level))
        if not nxt.any():
            break
        depth[nxt] = float(level)
        frontier = nxt.float()
    state["hop_depth"] = depth.cpu().numpy().astype(np.float32)

    z = np.load(OUT / "readout.npz")
    load = lambda k: torch.from_numpy(z[k]).to(dev)  # noqa: E731
    state["dec"] = {k: load(k) for k in ("W1", "b1", "W2", "b2", "mu", "sd")}
    state["downstream"] = load("downstream").long()
    bench = OUT / "results" / "benchmark.json"
    state["benchmark"] = _no_nan(json.loads(bench.read_text())) if bench.exists() else None
    mf = OUT / "samples" / "manifest.json"
    state["samples"] = json.loads(mf.read_text()) if mf.exists() else {"tiles": []}


def classify(img: Image.Image, sample: dict | None = None) -> dict:
    """Tile -> frozen circuit response -> damage score + connectome telemetry.

    One all-count pass covers the downstream feature rows and the brain-sample
    telemetry rows at once; a live region silence degrades both the score and
    the wave, reproducing the ablation in the browser.
    """
    d, sim = state["dec"], state["sim"]
    t0 = time.time()
    feats = retinal_sample(img)
    drive = build_drive(sim, feats[None])
    count_idx = torch.cat([state["downstream"], state["brain_idx_t"]])
    out = sim.run(drive, STEPS, count_idx=count_idx)
    counts = out["counts"][:, 0]
    nd = state["downstream"].numel()
    x = torch.log1p(counts[:nd]).unsqueeze(0)
    x = (x - d["mu"]) / d["sd"]
    h = torch.relu(x @ d["W1"].T + d["b1"])
    p_damaged = float(torch.softmax((h @ d["W2"].T + d["b2"])[0], dim=-1)[1])
    brain_counts = counts[nd:]
    act = torch.nonzero(brain_counts > 0).squeeze(1)
    k = min(140, act.numel())
    if k:
        tv, ti = torch.topk(brain_counts[act], k)
        active = [[int(act[i]), int(c)] for i, c in zip(ti.tolist(), tv.tolist())]
    else:
        active = []
    wave = out["region_wave"][:, ::6]
    resp = {
        "p_damaged": round(p_damaged, 4),
        "verdict": "damaged" if p_damaged >= 0.5 else "undamaged",
        "threshold": 0.5,
        "scores_calibrated": False,
        "spikes": float(out["total_spikes"].sum()),
        "sim_ms": round((time.time() - t0) * 1000, 1),
        "wave": [[round(float(wave[0, t]), 1), round(float(wave[1, t]), 1),
                  round(float(wave[2, t]), 1)] for t in range(wave.shape[1])],
        "active": active,
        "silenced": sim.silenced,
    }
    if sample is not None:
        resp["sample_id"] = sample["id"]
        resp["true_label"] = sample["true_label"]
        resp["fly_score_held_out"] = sample.get("fly_score")
    return resp


@app.get("/api/info")
def api_info():
    b = state["benchmark"]
    return {"available": state["dec"] is not None,
            "samples": len(state["samples"].get("tiles", [])),
            "benchmark_ready": b is not None,
            "meta": b["meta"] if b else None}


@app.get("/api/benchmark")
def api_benchmark():
    if state["benchmark"] is None:
        raise HTTPException(404, "benchmark not generated yet")
    return state["benchmark"]


@app.get("/api/samples")
def api_samples():
    return state["samples"]


@app.get("/api/sample/{sid}")
def api_sample(sid: str):
    for t in state["samples"].get("tiles", []):
        if t["id"] == sid:
            return FileResponse(OUT / "samples" / t["file"], media_type="image/jpeg")
    raise HTTPException(404, "unknown sample")


@app.post("/api/classify")
async def api_classify(request: Request):
    """Body is raw image bytes (any content type) or JSON {sample_id}."""
    ct = request.headers.get("content-type", "")
    sample = None
    if "application/json" in ct:
        sid = (await request.json()).get("sample_id")
        sample = next((t for t in state["samples"].get("tiles", []) if t["id"] == sid), None)
        if sample is None:
            raise HTTPException(404, "unknown sample")
        img = Image.open(OUT / "samples" / sample["file"]).convert("RGB")
    else:
        raw = await request.body()
        if not raw:
            raise HTTPException(409, "empty body: send image bytes or {\"sample_id\": ...}")
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(409, "not a decodable image")
    return await run_in_threadpool(classify, img, sample)


@app.post("/api/silence")
def api_silence(body: dict):
    """Silence a brain region live: 0 optic, 1 central, 2 VNC, null restores."""
    r = body.get("region")
    r = int(r) if r is not None and str(r) != "null" else None
    if r is not None and r not in (0, 1, 2):
        raise HTTPException(409, "region must be 0, 1 or 2")
    state["sim"].set_silence(r)
    return {"ok": True, "silenced": r}


@app.get("/api/silence")
def api_silence_get():
    return {"silenced": state["sim"].silenced}


@app.get("/api/brain")
def api_brain():
    b = state["brain"]
    dep = state["hop_depth"][b["idx"]]
    return JSONResponse({
        "points": [[round(float(x), 3), round(float(y), 3), round(float(zz), 3), int(rr),
                    (int(d) if d < 1e8 else -1)]
                   for (x, y, zz), rr, d in zip(b["points"], b["region"], dep)]})


@app.get("/healthz")
def healthz():
    return {"ok": True, "device": state["dev"], "decoder": state["dec"] is not None}


@app.middleware("http")
async def no_store_html(request, call_next):
    """HTML and API responses must never come from a stale browser cache."""
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p == "/index.html" or p.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
