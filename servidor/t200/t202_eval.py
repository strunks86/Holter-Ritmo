"""T202: prueba final en registros NO vistos: T158 vs T200 vs T201 vs T202.

Umbral por objetivo = max F1 en validacion limpia del mismo modelo. Por condicion (limpio y ruido
real de PRUEBA en 3 canales a 12/6/0/-6 dB y ruido puro): falsos positivos por 1000 ventanas
negativas y Se. En limpio ademas AUROC, VPP, Se exigiendo VPP 0,90 y 0,95, S falsas en ventanas
solo-V, V falsas en ventanas solo-S, y TSV falsa en taquicardia sinusal (PTB-XL STACH).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ruido  # noqa: E402
import t202_train as S  # noqa: E402

R, T, tt = S.R, S.T, S.tt
IDX = {"FA": 0, "FLUT": 1, "PVC": 2, "PAC": 3, "TV": 5, "TSV": 6}


class DSN(Dataset):
    def __init__(self, srcs, pairs, snr, puro=False):
        self.srcs, self.pairs, self.snr, self.puro = srcs, pairs, snr, puro

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, j = self.pairs[i]
        x = np.asarray(self.srcs[s].X[j], dtype=np.float32).T.copy()
        y = S.y7(self.srcs[s], j)
        if self.snr is not None or self.puro:
            rng = np.random.default_rng(7_000_000 + i)
            x, y = ruido.mezclar(x, y, R.banco()["test"], rng, snr=self.snr, todos=True, puro=self.puro)
        return torch.from_numpy(x), torch.from_numpy(y)


def thr_f1(y, s):
    if (y == 1).sum() == 0:
        return 0.0
    o = np.argsort(-s)
    ys = y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    return float(s[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])


def sub(pairs, y, cap, rng):
    raro = (y[:, 5] == 1) | (y[:, 6] == 1)
    keep, rest = np.where(raro)[0], np.where(~raro)[0]
    n = max(cap - len(keep), 0)
    if len(rest) > n:
        rest = rng.choice(rest, n, replace=False)
    ix = sorted(np.concatenate([keep, rest]))
    return [pairs[i] for i in ix]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="T158=/root/holter/t158/runs/full_s0/best.pt,T200=/root/holter/t200/runs/t200/best.pt,"
                                         "T201=/root/holter/t200/runs/noche/best.pt,T202=/root/holter/t200/runs/t202/best.pt")
    ap.add_argument("--tags", default="holter,icentia,taqui")
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    R.banco()
    srcs = S.load_sources7(args.tags.split(","))
    rng = np.random.default_rng(5)
    va = T.pairs_for(srcs, "val", regular_only=True)
    te = T.pairs_for(srcs, "test", regular_only=True)
    va = sub(va, np.stack([S.y7(srcs[s], j) for s, j in va]), args.cap, rng)
    te = sub(te, np.stack([S.y7(srcs[s], j) for s, j in te]), args.cap, rng)
    stach = np.array([srcs[s].TQ[j, 2] == 1 for s, j in te])
    mk = lambda ds: DataLoader(ds, batch_size=256, num_workers=8)  # noqa: E731
    conds = [("limpio", None, False), ("12dB", 12.0, False), ("6dB", 6.0, False), ("0dB", 0.0, False),
             ("-6dB", -6.0, False), ("ruido_puro", None, True)]
    res = {}
    for spec in args.modelos.split(","):
        name, path = spec.split("=")
        if not Path(path).exists():
            continue
        net, n0 = S.make_model7(dev, path)
        tg = [t for t, k in IDX.items() if k < n0]
        lv, yv = T.predict(net, mk(DSN(srcs, va, None)), dev)
        thr = {t: thr_f1(yv[yv[:, IDX[t]] >= 0, IDX[t]].astype(int), lv[yv[:, IDX[t]] >= 0, IDX[t]]) for t in tg}
        r = {"thr": thr, "n_salidas": n0}
        ruido_limpio = None
        for cname, snr, puro in conds:
            lt, yt = T.predict(net, mk(DSN(srcs, te, snr, puro)), dev)
            c = {}
            for t in tg:
                k = IDX[t]
                neg, pos = yt[:, k] == 0, yt[:, k] == 1
                pred = lt[:, k] >= thr[t]
                d = {"fp_por_1000": float(1000 * pred[neg].mean()) if neg.any() else None,
                     "se": float(pred[pos].mean()) if pos.any() else None, "n_neg": int(neg.sum()), "n_pos": int(pos.sum())}
                if cname == "limpio" and pos.any() and neg.any():
                    m = yt[:, k] >= 0
                    sc, yy = lt[m, k], yt[m, k]
                    d["auroc"] = tt.auroc(yy, sc)
                    d["ppv"] = float((pred & pos).sum() / max(pred.sum(), 1))
                    o = np.argsort(-sc)
                    tp = np.cumsum(yy[o] == 1)
                    prec = tp / np.arange(1, len(o) + 1)
                    for vpp in (0.90, 0.95):
                        ok = np.where(prec >= vpp)[0]
                        d[f"se_a_vpp{int(vpp * 100)}"] = float(tp[ok[-1]] / max((yy == 1).sum(), 1)) if len(ok) else 0.0
                c[t] = d
            v_sin_s = (yt[:, 2] == 1) & (yt[:, 3] == 0)
            s_sin_v = (yt[:, 3] == 1) & (yt[:, 2] == 0)
            if v_sin_s.any():
                c["S_falsa_en_solo_V_por_1000"] = float(1000 * (lt[v_sin_s, 3] >= thr["PAC"]).mean())
            if s_sin_v.any():
                c["V_falsa_en_solo_S_por_1000"] = float(1000 * (lt[s_sin_v, 2] >= thr["PVC"]).mean())
            if "TSV" in tg and cname == "limpio" and stach.any():
                c["TSV_falsa_en_taquicardia_sinusal_por_1000"] = float(1000 * (lt[stach, 6] >= thr["TSV"]).mean())
                c["n_taquicardia_sinusal"] = int(stach.sum())
            if n0 >= 5:
                if cname == "limpio":
                    ruido_limpio = lt[:, 4]
                    c["RUIDO_marcado_en_limpio"] = float((lt[:, 4] >= 0).mean())
                else:
                    lab = np.concatenate([np.zeros(len(ruido_limpio)), (yt[:, 4] == 1).astype(float)])
                    keep = np.concatenate([np.ones(len(ruido_limpio), bool), yt[:, 4] >= 0])
                    sc = np.concatenate([ruido_limpio, lt[:, 4]])
                    if (lab[keep] == 1).any():
                        c["RUIDO_auroc_vs_limpio"] = tt.auroc(lab[keep], sc[keep])
                    c["RUIDO_marcado"] = float((lt[:, 4] >= 0).mean())
            r[cname] = c
        res[name] = r
        del net
        torch.cuda.empty_cache()
    Path(f"/root/holter/t200/eval_{args.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = []
    for cname, _, _ in conds:
        lines.append(f"== {cname}")
        for name, r in res.items():
            c = r[cname]
            cells = []
            for t in IDX:
                if t not in c:
                    continue
                d = c[t]
                s = f"{t}: FP/1000 {d['fp_por_1000']:.1f}" if d["fp_por_1000"] is not None else f"{t}: -"
                if d.get("se") is not None:
                    s += f" Se {d['se']:.3f}"
                if "auroc" in d:
                    s += f" AUROC {d['auroc']:.4f} VPP {d['ppv']:.3f} Se@VPP90 {d['se_a_vpp90']:.3f} Se@VPP95 {d['se_a_vpp95']:.3f} (pos {d['n_pos']})"
                cells.append(s)
            extra = {k: round(v, 4) for k, v in c.items() if not isinstance(v, dict)}
            lines.append(f"{name}: " + " | ".join(cells) + f" | {extra}")
    txt = "\n".join(lines)
    Path(f"/root/holter/t200/eval_{args.name}.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
