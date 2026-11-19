"""T201: T200 + 5a salida RUIDO/ARTEFACTO entrenada con ruido real de NSTDB mezclado en 3 canales.

- Parte del mejor T200 (4 salidas); la cabeza se amplia a 5 (la fila RUIDO parte en cero).
- Aumentacion en linea (solo train): 35 % de las ventanas reciben ruido (ruido.mezclar), 3 % son
  ruido puro. Las ventanas sin aumentar llevan RUIDO = 0.
- Validacion fija: la mitad de val limpia y la otra mitad con ruido de PRUEBA (semilla por indice).
- Mismo esquema de tiempo, puntos de control y recarga en caliente que t200_train.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ruido  # noqa: E402
import t200_train as T  # noqa: E402

tt = T.tt
_spec = importlib.util.spec_from_file_location("t158_build", T.REPO / "scripts/t158_build.py")
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)
TARGETS5 = ["FA", "FLUT", "PVC", "PAC", "RUIDO"]
BANCO = None


def banco():
    global BANCO
    if BANCO is None:
        BANCO = ruido.cargar_banco(tb)
    return BANCO


class DS5(Dataset):
    def __init__(self, srcs, pairs, modo, p_aug=0.35, p_puro=0.03, seed=0):
        self.srcs, self.pairs, self.modo = srcs, pairs, modo
        self.p_aug, self.p_puro, self.seed = p_aug, p_puro, seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, j = self.pairs[i]
        x = np.asarray(self.srcs[s].X[j], dtype=np.float32).T.copy()
        y = np.concatenate([self.srcs[s].Y[j], [0.0]]).astype(np.float32)
        if y[1] == 1.0:
            y[0] = -1.0  # flutter: FA enmascarada, igual que T158 (no castigar FA en flutter)
        if self.modo == "train":
            rng = np.random.default_rng()
            u = rng.random()
            if u < self.p_puro:
                x, y = ruido.mezclar(x, y, banco()["train"], rng, puro=True)
            elif u < self.p_puro + self.p_aug:
                x, y = ruido.mezclar(x, y, banco()["train"], rng)
        elif self.modo == "val" and i % 2 == 1:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            x, y = ruido.mezclar(x, y, banco()["test"], rng)
        return torch.from_numpy(x), torch.from_numpy(y)


def avg_prec(y, s):
    o = np.argsort(-s)
    ys = y[o].astype(float)
    if ys.sum() == 0:
        return float("nan")
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def ppv_at_se(y, s, se=0.85):
    pos = np.sort(s[y == 1])[::-1]
    if len(pos) == 0:
        return float("nan")
    thr = pos[max(int(np.ceil(se * len(pos))) - 1, 0)]
    p = s >= thr
    return float((p & (y == 1)).sum() / max(p.sum(), 1))


def puntaje(lv, yv):
    """Criterio para elegir el mejor: precision promedio (castiga los falsos positivos), con
    la de S (PAC) contando doble porque Jorge no quiere S falsas."""
    aps = []
    for k in range(5):
        m = yv[:, k] >= 0
        aps.append(avg_prec(yv[m, k], lv[m, k]) if m.sum() else float("nan"))
    w = np.array([1, 0.5, 1, 2, 1.0])
    ok = ~np.isnan(aps)
    return float((np.array(aps)[ok] * w[ok]).sum() / w[ok].sum()), aps


def mean_auroc5(lv, yv):
    per = []
    for k in range(5):
        m = yv[:, k] >= 0
        per.append(tt.auroc(yv[m, k], lv[m, k]) if m.sum() else float("nan"))
    ok = [a for a in per if not np.isnan(a)]
    return float(np.mean(ok)) if ok else float("nan"), per


def make_model5(dev, init):
    net = tt.build_net(150)
    ck = torch.load(str(init), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    n_out = sd["dense.weight"].shape[0]
    net.dense = nn.Linear(1024, n_out)
    net.load_state_dict(sd)
    if n_out == 4:
        old = net.dense
        net.dense = nn.Linear(1024, 5)
        with torch.no_grad():
            net.dense.weight.zero_()
            net.dense.bias.fill_(-2.0)
            net.dense.weight[:4] = old.weight
            net.dense.bias[:4] = old.bias
    return net.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t201")
    ap.add_argument("--init", default="/root/holter/t200/runs/t200/best.pt")
    ap.add_argument("--tags", default="holter,icentia")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_bb", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_cap", type=int, default=30000)
    ap.add_argument("--paciencia", type=int, default=4, help="cortes sin mejorar antes de parar")
    args = ap.parse_args()
    torch.manual_seed(201)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(201)
    dev = torch.device("cuda:0")
    out = Path("/root/holter/t200/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "train.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    tags = args.tags.split(",")
    banco()
    log("banco de ruido NSTDB cargado: " + ", ".join(
        f"{t} {v.shape[0] / 500 / 60:.0f}+{banco()['test'][t].shape[0] / 500 / 60:.0f} min" for t, v in banco()["train"].items()))

    def build(srcs, make_val):
        tr = T.pairs_for(srcs, "train")
        dbs_tr = np.array([srcs[s].db[j] for s, j in tr])
        names, counts = np.unique(dbs_tr, return_counts=True)
        w_db = {n: 1.0 / math.sqrt(c) for n, c in zip(names, counts, strict=True)}
        weights = np.array([w_db[d] for d in dbs_tr])
        ytr = np.stack([srcs[s].Y[j] for s, j in tr])
        pos = ((ytr == 1) * weights[:, None]).sum(0)
        neg = ((ytr == 0) * weights[:, None]).sum(0)
        pw4 = np.clip(np.sqrt(np.maximum(neg, 1e-9) / np.maximum(pos, 1e-9)), 1.0, 30.0)
        pw4[3] = 1.0  # S: sin premio extra a los positivos -> menos S falsas
        pw = np.concatenate([pw4, [2.0]])
        log(f"train={len(tr)} bases={dict(zip(names.tolist(), counts.tolist(), strict=True))}")
        log(f"pos_weight={np.round(pw, 2).tolist()}")
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                        num_samples=10**8, replacement=True)
        dl = DataLoader(DS5(srcs, tr, "train"), batch_size=args.bs, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
        dl_va = None
        if make_val:
            va = T.pairs_for(srcs, "val", regular_only=True)
            if len(va) > args.val_cap:
                va = [va[i] for i in sorted(rng.choice(len(va), args.val_cap, replace=False))]
            log(f"val={len(va)} (mitad con ruido de prueba)")
            dl_va = DataLoader(DS5(srcs, va, "val", seed=201), batch_size=256,
                               num_workers=args.workers, pin_memory=True)
        return dl, dl_va, torch.tensor(pw, dtype=torch.float32, device=dev)

    def present():
        return [t for t in tags if (T.NEW / f"{t}_meta.npz").exists()]

    loaded = present()
    srcs = T.load_sources(loaded)
    log(f"fuentes: t158 + {loaded}")
    dl, dl_va, pw_t = build(srcs, True)

    net = make_model5(dev, args.init)
    head = list(net.dense.parameters())
    bb = [p for n, p in net.named_parameters() if not n.startswith("dense.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_bb}, {"params": head, "lr": args.lr_head}],
                            weight_decay=args.wd)
    base_lr = [args.lr_bb, args.lr_head]
    scaler = torch.amp.GradScaler("cuda")

    lv, yv = T.predict(net, dl_va, dev)
    _, per0 = mean_auroc5(lv, yv)
    mv0, aps0 = puntaje(lv, yv)
    m3 = yv[:, 3] >= 0
    log(f"punto de partida puntaje_AP={mv0:.4f} AP={[round(x, 4) for x in aps0]} "
        f"AUROC={[round(x, 4) for x in per0]} S_VPP@Se0.85={ppv_at_se(yv[m3, 3], lv[m3, 3]):.3f}")
    best = -1.0
    hist = [{"min": 0, "val_mean_auroc": mv0, "per": per0}]
    budget = args.minutes * 60
    t_start = time.time()
    next_eval = args.every * 60
    step, run_loss, nb = 0, 0.0, 0

    sin_mejora = [0]

    def evaluar(el):
        nonlocal best
        lv, yv = T.predict(net, dl_va, dev)
        ma, per = mean_auroc5(lv, yv)
        mv, aps = puntaje(lv, yv)
        m3 = yv[:, 3] >= 0
        pp = ppv_at_se(yv[m3, 3], lv[m3, 3], 0.85)
        hist.append({"min": round(el / 60, 1), "step": step, "puntaje_ap": mv, "auroc": per,
                     "ap": aps, "S_vpp_a_se085": pp})
        log(f"VAL min={el / 60:.1f} puntaje_AP={mv:.4f} AP={[round(x, 4) for x in aps]} "
            f"AUROC={[round(x, 4) for x in per]} S_VPP@Se0.85={pp:.3f}")
        ck = {"state_dict": net.state_dict(), "step": step, "val_mean_auroc": mv,
              "targets": TARGETS5, "base": str(args.init)}
        torch.save(ck, out / "last.pt")
        if mv > best + 1e-4:
            best = mv
            sin_mejora[0] = 0
            torch.save(ck, out / "best.pt")
            log("  nuevo mejor -> best.pt")
        else:
            sin_mejora[0] += 1
            log(f"  sin mejora ({sin_mejora[0]}/{args.paciencia})")
        (out / "hist.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")

    net.train()
    it = iter(dl)
    while True:
        el = time.time() - t_start
        if el >= budget:
            break
        xb, yb = next(it)
        frac = el / budget
        mult = min(1.0, frac / 0.03) * 0.5 * (1 + math.cos(math.pi * frac))
        for g, b in zip(opt.param_groups, base_lr, strict=True):
            g["lr"] = b * max(mult, 0.02)
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(tt.to12(xb))
        logits = logits.float()
        m = (yb >= 0).float()
        elem = nn.functional.binary_cross_entropy_with_logits(
            logits, yb.clamp(min=0), pos_weight=pw_t, reduction="none")
        loss = (elem * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        nb += 1
        run_loss += loss.item()
        if step % 500 == 0:
            log(f"paso {step} min={el / 60:.1f} loss={run_loss / nb:.4f} "
                f"{step * args.bs / el:.0f} ventanas/s lr_mult={mult:.3f}")
            run_loss, nb = 0.0, 0
        if el >= next_eval:
            next_eval += args.every * 60
            evaluar(el)
            if sin_mejora[0] >= args.paciencia:
                log("PARADA: ya no mejora")
                break
            if present() != loaded and el < budget - 10 * 60:
                loaded = present()
                log(f"llegaron datos nuevos: {loaded}; recargo")
                del it, dl
                srcs_new = T.load_sources(loaded)
                dl, _, pw_t = build(srcs_new, False)
                it = iter(dl)
            net.train()
    evaluar(time.time() - t_start)
    log(f"FIN pasos={step} mejor_val={best:.4f} (partida={mv0:.4f})")
    os._exit(0)


if __name__ == "__main__":
    main()
