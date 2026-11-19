"""T200: sigue entrenando T158 (ECGFounder ajustado, 3 canales) con las bases Holter nuevas.

- Parte de los pesos de T158 (runs/full_s0/best.pt), modelo completo ('full').
- Datos: ventanas de T158 (PTB-XL, CPSC2018, AFDB, LTAFDB) + ventanas nuevas de t200_build
  (MIT-BIH, SVDB, INCART, LTDB, EDB, SDDB, SHDB-AF, CPSC2021, Icentia11k).
- Muestreo ponderado por base (peso ~ 1/sqrt(n_base)) para que Icentia no lo tape todo.
- Entrena por TIEMPO (--minutes); tasa de aprendizaje con calentamiento y coseno sobre el tiempo.
- Cada --every minutos: AUROC en validacion (registros no vistos), guarda last.pt y best.pt.
"""
from __future__ import annotations

import argparse
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

REPO = Path(os.environ.get("HOLTERAI", "/root/holter/holterai"))
sys.path.insert(0, str(REPO / "scripts"))
import t158_train as tt  # noqa: E402

OLD = Path("/root/holter/t158/data")
NEW = Path("/root/holter/t200/data")
T158_PT = Path("/root/holter/t158/runs/full_s0/best.pt")


class Src:
    def __init__(self, xpath, mpath):
        self.X = np.load(xpath, mmap_mode="r")
        m = np.load(mpath)
        self.db, self.split, self.Y = m["db"], m["split"], m["Y"].astype(np.float32)
        self.start = m["start"]
        self.rec = m["rec"] if "rec" in m else np.array([""] * len(self.db))


def load_sources(tags):
    srcs = []
    if (OLD / "windows_X.npy").exists():
        srcs.append(Src(OLD / "windows_X.npy", OLD / "windows_meta.npz"))
    for t in tags:
        if (NEW / f"{t}_X.npy").exists():
            srcs.append(Src(NEW / f"{t}_X.npy", NEW / f"{t}_meta.npz"))
    return srcs


class DS(Dataset):
    def __init__(self, srcs, pairs):
        self.srcs, self.pairs = srcs, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, j = self.pairs[i]
        x = torch.from_numpy(np.asarray(self.srcs[s].X[j], dtype=np.float32)).T.contiguous()
        return x, torch.from_numpy(self.srcs[s].Y[j])


def pairs_for(srcs, split, regular_only=False):
    out = []
    for s, src in enumerate(srcs):
        idx = np.where(src.split == split)[0]
        if regular_only:
            keep = ~((src.db[idx] == "ltafdb") & (np.round(src.start[idx]) % 120 != 0))
            idx = idx[keep]
        out += [(s, int(j)) for j in idx]
    return out


def make_model(dev, init=None):
    net = tt.build_net(150)
    net.dense = nn.Linear(1024, 4)
    ck = torch.load(str(init or T158_PT), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev)


def predict(net, loader, dev):
    net.eval()
    ls, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for xb, yb in loader:
            ls.append(net(tt.to12(xb.to(dev, non_blocking=True))).float().cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ls), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t200")
    ap.add_argument("--tags", default="new")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_bb", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_cap", type=int, default=30000)
    ap.add_argument("--init", default=str(T158_PT))
    args = ap.parse_args()
    torch.manual_seed(200)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(200)
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

    def build(srcs):
        tr = pairs_for(srcs, "train")
        va = pairs_for(srcs, "val", regular_only=True)
        if len(va) > args.val_cap:
            va = [va[i] for i in sorted(rng.choice(len(va), args.val_cap, replace=False))]
        dbs_tr = np.array([srcs[s].db[j] for s, j in tr])
        names, counts = np.unique(dbs_tr, return_counts=True)
        w_db = {n: 1.0 / math.sqrt(c) for n, c in zip(names, counts, strict=True)}
        weights = np.array([w_db[d] for d in dbs_tr])
        ytr = np.stack([srcs[s].Y[j] for s, j in tr])
        pos = ((ytr == 1) * weights[:, None]).sum(0)
        neg = ((ytr == 0) * weights[:, None]).sum(0)
        pw = np.clip(np.sqrt(np.maximum(neg, 1e-9) / np.maximum(pos, 1e-9)), 1.0, 30.0)
        log(f"train={len(tr)} val={len(va)} bases={dict(zip(names.tolist(), counts.tolist(), strict=True))}")
        log(f"pos_weight={np.round(pw, 2).tolist()}")
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                        num_samples=10**8, replacement=True)
        dl = DataLoader(DS(srcs, tr), batch_size=args.bs, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=False, prefetch_factor=4)
        dl_va = DataLoader(DS(srcs, va), batch_size=256, num_workers=args.workers, pin_memory=True)
        return dl, dl_va, torch.tensor(pw, dtype=torch.float32, device=dev)

    def present():
        return [t for t in tags if (NEW / f"{t}_meta.npz").exists()]

    loaded = present()
    srcs = load_sources(loaded)
    log(f"fuentes: t158 + {loaded}")
    dl, dl_va, pw_t = build(srcs)

    net = make_model(dev, args.init)
    head = list(net.dense.parameters())
    bb = [p for n, p in net.named_parameters() if not n.startswith("dense.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_bb}, {"params": head, "lr": args.lr_head}],
                            weight_decay=args.wd)
    base_lr = [args.lr_bb, args.lr_head]
    scaler = torch.amp.GradScaler("cuda")

    lv, yv = predict(net, dl_va, dev)
    mv0, per0 = tt.mean_val_auroc(lv, yv)
    log(f"T158 punto de partida val_mean_auroc={mv0:.4f} per={[round(x, 4) for x in per0]}")
    best = mv0
    hist = [{"min": 0, "val_mean_auroc": mv0, "per": per0}]
    budget = args.minutes * 60
    t_start = time.time()
    next_eval = args.every * 60
    step, run_loss, nb = 0, 0.0, 0
    net.train()
    it = iter(dl)
    while True:
        xb, yb = next(it)
        el = time.time() - t_start
        if el >= budget:
            break
        frac = el / budget
        mult = min(1.0, frac / 0.05) * 0.5 * (1 + math.cos(math.pi * frac))
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
        run_loss += float(loss)
        if step % 200 == 0:
            log(f"paso {step} min={el / 60:.1f} loss={run_loss / nb:.4f} "
                f"{step * args.bs / el:.0f} ventanas/s lr_mult={mult:.3f}")
        if el >= next_eval:
            next_eval += args.every * 60
            lv, yv = predict(net, dl_va, dev)
            mv, per = tt.mean_val_auroc(lv, yv)
            hist.append({"min": round(el / 60, 1), "step": step, "loss": run_loss / nb,
                         "val_mean_auroc": mv, "per": per})
            log(f"VAL min={el / 60:.1f} val_mean_auroc={mv:.4f} per={[round(x, 4) for x in per]}")
            ck = {"state_dict": net.state_dict(), "step": step, "val_mean_auroc": mv,
                  "targets": tt.TARGETS, "base": "t158_full_s0"}
            torch.save(ck, out / "last.pt")
            if mv > best:
                best = mv
                torch.save(ck, out / "best.pt")
                log("  nuevo mejor -> best.pt")
            run_loss, nb = 0.0, 0
            if present() != loaded and el < budget - 5 * 60:
                loaded = present()
                log(f"llegaron datos nuevos: {loaded}; recargo")
                del it, dl
                srcs = load_sources(loaded)
                dl, dl_va_new, pw_t = build(srcs)
                it = iter(dl)
            net.train()
    lv, yv = predict(net, dl_va, dev)
    mv, per = tt.mean_val_auroc(lv, yv)
    el = time.time() - t_start
    hist.append({"min": round(el / 60, 1), "step": step, "val_mean_auroc": mv, "per": per})
    log(f"VAL FINAL min={el / 60:.1f} val_mean_auroc={mv:.4f} per={[round(x, 4) for x in per]}")
    ck = {"state_dict": net.state_dict(), "step": step, "val_mean_auroc": mv,
          "targets": tt.TARGETS, "base": "t158_full_s0"}
    torch.save(ck, out / "last.pt")
    if mv > best:
        torch.save(ck, out / "best.pt")
        best = mv
    (out / "hist.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    log(f"FIN pasos={step} mejor_val={best:.4f} (T158={mv0:.4f})")
    os._exit(0)


if __name__ == "__main__":
    main()
