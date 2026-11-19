"""T200: compara T158 contra el modelo nuevo en registros no vistos (split 'test').

Por modelo y objetivo (FA, FLUT, PVC, PAC):
  - AUROC en test.
  - Umbral = max F1 en VALIDACION (del mismo modelo); Se y VPP en test con ese umbral.
  - VPP del nuevo a la MISMA Se que logra T158 en test (comparacion justa a Se fija).
Desglose por base. Dos conjuntos: 'viejas' (test de T158) y 'nuevas' (test de las bases nuevas).
Salida: /root/holter/t200/eval_<name>.json y un resumen en texto.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t200_train as T  # noqa: E402

tt = T.tt


def load(path, dev):
    net = tt.build_net(150)
    net.dense = nn.Linear(1024, 4)
    net.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"])
    return net.to(dev)


def thr_f1(y, s):
    o = np.argsort(-s)
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + ys.sum())
    k = int(np.argmax(f1))
    return float(s[o][k])


def se_ppv(y, s, thr):
    p = s >= thr
    tp = float((p & (y == 1)).sum())
    return tp / max((y == 1).sum(), 1), tp / max(p.sum(), 1)


def ppv_at_se(y, s, se_target):
    pos = np.sort(s[y == 1])[::-1]
    if len(pos) == 0:
        return float("nan")
    k = max(int(np.ceil(se_target * len(pos))) - 1, 0)
    return se_ppv(y, s, pos[k])[1]


def metrics(lt, yt, lvv, yvv, ref_se=None):
    res = {}
    for k, t in enumerate(tt.TARGETS):
        m, mv = yt[:, k] >= 0, yvv[:, k] >= 0
        y, s = yt[m, k].astype(int), lt[m, k]
        if m.sum() == 0 or y.sum() == 0 or y.sum() == len(y):
            res[t] = {"n": int(m.sum()), "pos": int(y.sum())}
            continue
        thr = thr_f1(yvv[mv, k].astype(int), lvv[mv, k]) if (yvv[mv, k] == 1).any() else 0.0
        se, ppv = se_ppv(y, s, thr)
        r = {"n": int(m.sum()), "pos": int(y.sum()), "auroc": tt.auroc(y, s), "se": se, "ppv": ppv}
        if ref_se and t in ref_se and "se" in ref_se[t]:
            r["ppv_a_se_t158"] = ppv_at_se(y, s, ref_se[t]["se"])
        res[t] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="/root/holter/t200/runs/t200/best.pt")
    ap.add_argument("--name", default="t200")
    ap.add_argument("--tags", default="new")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    srcs = T.load_sources(args.tags.split(","))
    va = T.pairs_for(srcs, "val", regular_only=True)
    te = T.pairs_for(srcs, "test", regular_only=True)
    mk = lambda p: DataLoader(T.DS(srcs, p), batch_size=256, num_workers=12)  # noqa: E731
    db_te = np.array([srcs[s].db[j] for s, j in te])
    db_va = np.array([srcs[s].db[j] for s, j in va])
    old = {"ptbxl", "cpsc2018", "afdb", "ltafdb"}
    out = {}
    preds = {}
    for name, path in (("T158", T.T158_PT), ("nuevo", Path(args.cand))):
        net = load(path, dev)
        lv, yv = T.predict(net, mk(va), dev)
        lt, yt = T.predict(net, mk(te), dev)
        preds[name] = (lv, yv, lt, yt)
        del net
        torch.cuda.empty_cache()
    for name in ("T158", "nuevo"):
        lv, yv, lt, yt = preds[name]
        out[name] = {}
        for grp, sel_t, sel_v in (
                ("viejas", np.isin(db_te, list(old)), np.isin(db_va, list(old))),
                ("nuevas", ~np.isin(db_te, list(old)), ~np.isin(db_va, list(old))),
                ("todas", np.ones(len(te), bool), np.ones(len(va), bool))):
            ref = out["T158"][grp] if name == "nuevo" else None
            out[name][grp] = metrics(lt[sel_t], yt[sel_t], lv[sel_v], yv[sel_v], ref)
        out[name]["por_base"] = {}
        for d in sorted(set(db_te)):
            st, sv = db_te == d, db_va == d
            if sv.sum() == 0:
                sv = np.ones(len(va), bool)
            ref = out["T158"]["por_base"].get(d) if name == "nuevo" else None
            out[name]["por_base"][d] = metrics(lt[st], yt[st], lv[sv], yv[sv], ref)
    p = Path(f"/root/holter/t200/eval_{args.name}.json")
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = []
    for grp in ("viejas", "nuevas", "todas"):
        lines.append(f"== {grp}")
        for t in tt.TARGETS:
            a, b = out["T158"][grp].get(t, {}), out["nuevo"][grp].get(t, {})
            if "auroc" not in a or "auroc" not in b:
                continue
            lines.append(f"{t:5s} n={a['n']} pos={a['pos']} | T158 AUROC {a['auroc']:.4f} Se {a['se']:.3f} VPP {a['ppv']:.3f}"
                         f" | nuevo AUROC {b['auroc']:.4f} Se {b['se']:.3f} VPP {b['ppv']:.3f}"
                         f" VPP@SeT158 {b.get('ppv_a_se_t158', float('nan')):.3f}")
    lines.append("== por base (AUROC T158 -> nuevo)")
    for d, r in out["nuevo"]["por_base"].items():
        cells = []
        for t in tt.TARGETS:
            a = out["T158"]["por_base"][d].get(t, {})
            if "auroc" in a and "auroc" in r.get(t, {}):
                cells.append(f"{t} {a['auroc']:.3f}->{r[t]['auroc']:.3f} (pos {a['pos']})")
        lines.append(f"{d}: " + "; ".join(cells))
    txt = "\n".join(lines)
    Path(f"/root/holter/t200/eval_{args.name}.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
