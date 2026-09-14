"""MaleCNS v1.0 import -> CSR graph for the firefly experiment.

Data: male Drosophila melanogaster CNS v1.0 (HHMI Janelia + Google Research,
CC-BY), flat-connectome feathers. Node/edge policy follows nftechie/doomfly
(MIT): every entry with an assigned superclass except Glia; all released edges
at minconf 0.5; weights are synapse contact counts. GABA synapses are treated
as inhibitory (modeling choice); everything else excitatory. The wiring is
never modified or trained -- it is used purely as a frozen feature extractor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path("/library/datasets/malecns_v1")
CACHE = DATA / "processed-firefly"

F_ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_NT = "body-neurotransmitters-male-cns-v1.0.feather"
F_EDGES = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

N_CELLS = 9           # tic-tac-toe cells
N_SENS_PER_CELL = 6   # R1-R6 photoreceptors per board cell (own channel)
N_R8_PER_CELL = 5     # R8 cells per board cell (opponent channel)
N_MOTOR = 1024        # 512 descending + 512 VNC motor neurons


def _exact_ids(values: np.ndarray) -> np.ndarray:
    """bodyIds must stay exact integers (never through float)."""
    items = np.asarray(values)
    if items.dtype.kind == "f":
        raise ValueError("neuron IDs arrived as floats")
    if items.dtype.kind in "iu":
        return items.astype(np.uint64)
    return np.asarray([str(v) for v in items], dtype=np.uint64)


def build(cache_dir: Path = CACHE, data_dir: Path = DATA) -> dict:
    import pyarrow.feather as feather

    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    npz_path = cache_dir / "graph.npz"
    if meta_path.exists() and npz_path.exists():
        return json.loads(meta_path.read_text())

    cache_dir.mkdir(parents=True, exist_ok=True)

    ann = feather.read_table(data_dir / F_ANN).to_pandas()
    nt = feather.read_table(data_dir / F_NT).to_pandas().set_index("body")

    retain = ann.superclass.notna() & ann.superclass.astype(str).ne("") & ~ann.status.eq("Glia")
    nodes = ann.loc[retain].sort_values("bodyId").reset_index(drop=True)
    ids = _exact_ids(nodes.bodyId.to_numpy())
    n_nodes = len(ids)

    # per-neuron synaptic sign: GABA inhibitory, others excitatory (modeling choice)
    cons = nt.consensus_nt.reindex(nodes.bodyId.to_numpy())
    signs = np.where(cons.astype(str).str.lower().eq("gaba"), -1.0, 1.0).astype(np.float32)

    edges = feather.read_table(data_dir / F_EDGES)
    pre_all = _exact_ids(edges.column("body_pre").to_numpy(zero_copy_only=False))
    post_all = _exact_ids(edges.column("body_post").to_numpy(zero_copy_only=False))
    w_all = edges.column("weight").to_numpy(zero_copy_only=False).astype(np.float64)

    i = np.searchsorted(ids, pre_all)
    j = np.searchsorted(ids, post_all)
    keep = (i < n_nodes) & (j < n_nodes)
    keep &= ids[np.minimum(i, n_nodes - 1)] == pre_all
    keep &= ids[np.minimum(j, n_nodes - 1)] == post_all
    i, j, w = i[keep].astype(np.int64), j[keep].astype(np.int64), w_all[keep].astype(np.float64)

    outdeg = np.bincount(i, minlength=n_nodes).astype(np.float64)
    indeg = np.bincount(j, minlength=n_nodes).astype(np.float64)
    # strength = signed log contact count, shrunk by fan-out so hubs do not blow up
    val = (np.log1p(w) / np.sqrt(outdeg[i] + 1.0)) * signs[i]

    order = np.lexsort((j, i))  # CSR row-major ordering
    i, j, val = i[order], j[order], val[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(indptr, i + 1, 1)
    np.cumsum(indptr, out=indptr)

    # ---- deterministic task interface ------------------------------------
    # own channel: patches of R1-R6 spread across the whole retina
    r_all = nodes.index[(nodes.superclass == "ol_sensory") & (nodes.type == "R1-R6")]
    r_ids = nodes.bodyId.to_numpy()[r_all]
    own_pick = np.linspace(0, len(r_ids) - 1, N_CELLS * N_SENS_PER_CELL).astype(int)
    own_sens = np.searchsorted(ids, _exact_ids(r_ids[own_pick])).astype(np.int64)
    # opponent channel: R8 patches (color channel, cf. doomfly brightness/color split)
    r8_all = nodes.index[(nodes.superclass == "ol_sensory")
                         & nodes.type.astype(str).str.startswith("R8")]
    r8_ids = nodes.bodyId.to_numpy()[r8_all]
    opp_pick = np.linspace(0, len(r8_ids) - 1, N_CELLS * N_R8_PER_CELL).astype(int)
    opp_sens = np.searchsorted(ids, _exact_ids(r8_ids[opp_pick])).astype(np.int64)
    sensory_idx = np.concatenate([own_sens, opp_sens])

    # motor out: strongest descending neurons + strongest VNC motor neurons
    dn = nodes.index[nodes.superclass == "descending_neuron"]
    dn_ids = nodes.bodyId.to_numpy()[dn]
    dn_out = outdeg[np.searchsorted(ids, _exact_ids(dn_ids))]
    dn_rank = np.lexsort((dn_ids, -dn_out))[: N_MOTOR // 2]
    vm = nodes.index[nodes.superclass == "vnc_motor"]
    vm_ids = nodes.bodyId.to_numpy()[vm]
    vm_out = outdeg[np.searchsorted(ids, _exact_ids(vm_ids))]
    vm_rank = np.lexsort((vm_ids, -vm_out))[: N_MOTOR // 2]
    motor_idx = np.concatenate([
        np.searchsorted(ids, _exact_ids(dn_ids[dn_rank])),
        np.searchsorted(ids, _exact_ids(vm_ids[vm_rank])),
    ]).astype(np.int64)
    motor_types = (
        [str(t) if t is not None else "" for t in nodes.type.to_numpy()[dn][dn_rank]]
        + [str(t) if t is not None else "" for t in nodes.type.to_numpy()[vm][vm_rank]]
    )

    sup = nodes.superclass.astype(str).fillna("")
    region = np.select(
        [sup.str.startswith("ol_") | sup.str.startswith("visual_"),
         sup.str.startswith("cb_") | sup.isin(
             ["ascending_neuron", "sensory_ascending", "ENS", "descending_neuron",
              "cb_endocrine", "cb_motor"])],
        [0, 1], default=2).astype(np.uint8)   # 0 optic, 1 central, 2 VNC

    ppl = nodes.index[nodes.type == "PPL101"]
    ppl_idx = np.searchsorted(ids, _exact_ids(nodes.bodyId.to_numpy()[ppl])).astype(np.int64)

    np.savez_compressed(
        npz_path,
        indptr=indptr,
        indices=j.astype(np.int32),
        values=val.astype(np.float32),
        theta_scale=indeg.astype(np.float32),
        region=region,
        sensory_idx=sensory_idx,
        motor_idx=motor_idx,
        ppl_idx=ppl_idx,
    )
    meta = {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(i)),
        "gaba_fraction": float(np.mean(signs < 0)),
        "synapses_total": float(w_all.sum()),
        "synapses_retained": float(w.sum()),
        "sensory_idx": sensory_idx.tolist(),
        "motor_idx": motor_idx.tolist(),
        "motor_types": motor_types,
        "ppl_idx": ppl_idx.tolist(),
        "source_files": {"annotations": F_ANN, "nt": F_NT, "edges": F_EDGES},
        "interface": {
            "sensory": "own pieces -> 6-cell R1-R6 retinal patches (9x6); "
                       "opponent pieces -> 5-cell R8 patches (9x5)",
            "motor": "top 512 descending + top 512 VNC motor neurons by outgoing synapses",
            "dopamine": "PPL101 pair, aversive pulse on loss",
            "signs": "GABA edges negative; all other synapses positive",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_graph(cache_dir: Path = CACHE) -> dict:
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    meta["arrays"] = np.load(Path(cache_dir) / "graph.npz")
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("n_nodes", "n_edges", "gaba_fraction", "synapses_retained")}, indent=2))
