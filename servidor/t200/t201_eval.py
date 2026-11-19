"""T201: prueba de ruido y de limpio en registros NO vistos (split test), T158 vs T200 vs T201.

- Umbral de cada objetivo = max F1 en validacion LIMPIA del mismo modelo.
- Limpio: AUROC, Se y VPP por objetivo.
- Con ruido real de PRUEBA (NSTDB, minutos nunca usados en train) en los 3 canales a 12, 6, 0 y
  -6 dB, y ruido puro: tasa de falsos positivos (ventanas sin V marcadas como V, idem S y FA) y,
  para T201, deteccion de RUIDO (AUROC ruido <=6 dB vs limpio).
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
import t201_train as R  # noqa: E402

T, tt = R.T, R.tt
TG = ["FA", "FLUT", "PVC", "PAC"]


class DSN(Dataset):
    def __init__(self, srcs, pairs, snr, puro=False):
        self.srcs, self.pairs, self.snr, self.puro = srcs, pairs, snr, puro

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, j = self.pairs[i]
        x = np.asarray(self.srcs[s].X[j], dtype=np.float32).T.copy()
        y = np.concatenate([self.srcs[s].Y[j], [0.0]]).astype(np.float32)
        if y[1] == 1.0:
            y[0] = -1.0  # flutter: FA enmascarada, igual que T158 (no castigar FA en flutter)
        if self.snr is not None or self.puro:
            rng = np.random.default_rng(7_000_000 + i)
            x, y = ruido.mezclar(x, y, R.banco()["test"], rng, snr=self.snr, todos=True, puro=self.puro)
        return torch.from_numpy(x), torch.from_numpy(y)


def thr_f1(y, s):
    o = np.argsort(-s)
    ys = y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + ys.sum())
    return float(s[o][int(np.argmax(f1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t200", default="/root/holter/t200/runs/t200/best.pt")
    ap.add_argument("--t201", default="/root/holter/t200/runs/noche/best.pt")
    ap.add_argument("--tags", default="holter,icentia")
    ap.add_argument("--cap", type=int, default=25000)
    ap.add_argument("--name", default="ruido")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    R.banco()
    srcs = T.load_sources(args.tags.split(","))
    rng = np.random.default_rng(5)
    va = T.pairs_for(srcs, "val", regular_only=True)
    te = T.pairs_for(srcs, "test", regular_only=True)
    if len(va) > args.cap:
        va = [va[i] for i in sorted(rng.choice(len(va), args.cap, replace=False))]
    if len(te) > args.cap:
        te = [te[i] for i in sorted(rng.choice(len(te), args.cap, replace=False))]
    mk = lambda ds: DataLoader(ds, batch_size=256, num_workers=8)  # noqa: E731
    conds = [("limpio", None, False), ("12dB", 12.0, False), ("6dB", 6.0, False),
             ("0dB", 0.0, False), ("-6dB", -6.0, False), ("ruido_puro", None, True)]
    modelos = [("T158", T.T158_PT), ("T200", Path(args.t200)), ("T201", Path(args.t201))]
    res = {}
    for name, path in modelos:
        if not path.exists():
            continue
        net = R.make_model5(dev, path)
        n_out = 5
        lv, yv = T.predict(net, mk(DSN(srcs, va, None)), dev)
        thr = [thr_f1(yv[yv[:, k] >= 0, k].astype(int), lv[yv[:, k] >= 0, k]) for k in range(4)]
        r = {"thr": thr}
        clean_ruido = None
        for cname, snr, puro in conds:
            lt, yt = T.predict(net, mk(DSN(srcs, te, snr, puro)), dev)
            c = {}
            for k, t in enumerate(TG):
                neg = yt[:, k] == 0
                pos = yt[:, k] == 1
                pred = lt[:, k] >= thr[k]
                c[t] = {"fp_por_1000": float(1000 * pred[neg].mean()) if neg.any() else None,
                        "se": float(pred[pos].mean()) if pos.any() else None,
                        "n_neg": int(neg.sum()), "n_pos": int(pos.sum())}
                if cname == "limpio" and pos.any() and neg.any():
                    m = yt[:, k] >= 0
                    c[t]["auroc"] = tt.auroc(yt[m, k], lt[m, k])
                    c[t]["ppv"] = float((pred & pos).sum() / max(pred.sum(), 1))
                    sc, yy = lt[m, k], yt[m, k]
                    o = np.argsort(-sc)
                    tp = np.cumsum(yy[o] == 1)
                    prec = tp / np.arange(1, len(o) + 1)
                    for vpp in (0.90, 0.95):
                        ok = np.where(prec >= vpp)[0]
                        c[t][f"se_a_vpp{int(vpp * 100)}"] = float(tp[ok[-1]] / max((yy == 1).sum(), 1)) if len(ok) else 0.0
            v_sin_s = (yt[:, 2] == 1) & (yt[:, 3] == 0)
            s_sin_v = (yt[:, 3] == 1) & (yt[:, 2] == 0)
            if v_sin_s.any():
                c["S_falsa_en_ventanas_solo_V_por_1000"] = float(1000 * (lt[v_sin_s, 3] >= thr[3]).mean())
            if s_sin_v.any():
                c["V_falsa_en_ventanas_solo_S_por_1000"] = float(1000 * (lt[s_sin_v, 2] >= thr[2]).mean())
            if name == "T201":
                if cname == "limpio":
                    clean_ruido = lt[:, 4]
                    c["RUIDO_marcado_limpio"] = float((lt[:, 4] >= 0).mean())
                else:
                    sc = np.concatenate([clean_ruido, lt[:, 4]])
                    lab = np.concatenate([np.zeros(len(clean_ruido)), (yt[:, 4] == 1).astype(float)])
                    keep = np.concatenate([np.ones(len(clean_ruido), bool), yt[:, 4] >= 0])
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
        for name in res:
            c = res[name][cname]
            cells = []
            for t in TG:
                d = c[t]
                s = f"{t}: FP/1000 {d['fp_por_1000']:.1f}" if d["fp_por_1000"] is not None else f"{t}: -"
                if d.get("se") is not None:
                    s += f" Se {d['se']:.3f}"
                if "auroc" in d:
                    s += (f" AUROC {d['auroc']:.4f} VPP {d['ppv']:.3f} Se@VPP90 {d['se_a_vpp90']:.3f}"
                          f" Se@VPP95 {d['se_a_vpp95']:.3f}")
                cells.append(s)
            extra = {k: round(v, 4) for k, v in c.items() if k.startswith(("RUIDO", "S_falsa", "V_falsa"))}
            lines.append(f"{name}: " + " | ".join(cells) + (f" | {extra}" if extra else ""))
    txt = "\n".join(lines)
    Path(f"/root/holter/t200/eval_{args.name}.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
