"""T202: T201 + salidas TV y TSV (7 salidas: FA, FLUT, PVC, PAC, RUIDO, TV, TSV).

Parte del mejor T201 (5 salidas); la cabeza se amplia a 7. Datos: todo lo anterior + ventanas
'taqui' centradas en episodios; etiquetas TV/TSV de t202_labels (<tag>_tq.npz). Las ventanas con
TV o TSV se muestrean 8 veces mas. Mejor modelo = precision promedio ponderada (S cuenta doble).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ruido  # noqa: E402
import t201_train as R  # noqa: E402

T, tt = R.T, R.tt
TARGETS7 = ["FA", "FLUT", "PVC", "PAC", "RUIDO", "TV", "TSV"]
PESO = np.array([1, 0.5, 1, 2, 1, 1, 1.0])


def load_sources7(tags):
    srcs = []
    cands = [(T.OLD / "windows_X.npy", T.OLD / "windows_meta.npz", T.OLD / "windows_tq.npz")]
    cands += [(T.NEW / f"{t}_X.npy", T.NEW / f"{t}_meta.npz", T.NEW / f"{t}_tq.npz") for t in tags]
    for x, m, q in cands:
        if x.exists() and m.exists() and q.exists():
            s = T.Src(x, m)
            s.TQ = np.load(q)["TQ"]
            assert len(s.TQ) == len(s.Y)
            srcs.append(s)
    return srcs


def y7(src, j):
    y = np.empty(7, np.float32)
    y[:4] = src.Y[j]
    y[4] = src.TQ[j, 3] if src.TQ.shape[1] > 3 else 0.0  # RUIDO real (bases con calidad anotada)
    y[5:] = src.TQ[j, :2]
    if y[1] == 1.0:
        y[0] = -1.0
    return y


def perturbar(x, rng, p_drop=0.15):
    """Robustez a la colocacion real de electrodos: reordena los canales entre las 3 ranuras,
    invierte la polaridad de cada canal al azar y a veces apaga un canal (si hay >=2)."""
    x = x[rng.permutation(3)].copy()
    used = [j for j in range(3) if np.abs(x[j]).max() > 0]
    for j in used:
        if rng.random() < 0.5:
            x[j] = -x[j]
    if len(used) >= 2 and rng.random() < p_drop:
        x[used[rng.integers(0, len(used))]] = 0.0
    return x


class DS7(Dataset):
    def __init__(self, srcs, pairs, modo, p_aug=0.35, p_puro=0.03, seed=0, rob=0.0):
        self.srcs, self.pairs, self.modo = srcs, pairs, modo
        self.p_aug, self.p_puro, self.seed, self.rob = p_aug, p_puro, seed, rob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, j = self.pairs[i]
        x = np.asarray(self.srcs[s].X[j], dtype=np.float32).T.copy()
        y = y7(self.srcs[s], j)
        if self.modo == "valrob":
            x = perturbar(x, np.random.default_rng(self.seed * 7_000_003 + i), p_drop=0.0)
            if i % 2 == 1 and y[4] == 0.0:
                x, y = ruido.mezclar(x, y, R.banco()["test"], np.random.default_rng(self.seed * 1_000_003 + i))
        if self.modo == "train":
            rng = np.random.default_rng()
            if self.rob and rng.random() < self.rob:
                x = perturbar(x, rng)
            u = rng.random() if y[4] == 0.0 else 1.0  # no mezclar ruido sobre ruido real
            if u < self.p_puro:
                x, y = ruido.mezclar(x, y, R.banco()["train"], rng, puro=True)
            elif u < self.p_puro + self.p_aug:
                x, y = ruido.mezclar(x, y, R.banco()["train"], rng)
        elif self.modo == "val" and i % 2 == 1 and y[4] == 0.0:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            x, y = ruido.mezclar(x, y, R.banco()["test"], rng)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_model7(dev, init):
    net = tt.build_net(150)
    sd = torch.load(str(init), map_location="cpu", weights_only=False)["state_dict"]
    n_out = sd["dense.weight"].shape[0]
    net.dense = nn.Linear(1024, n_out)
    net.load_state_dict(sd)
    if n_out < 7:
        old = net.dense
        net.dense = nn.Linear(1024, 7)
        with torch.no_grad():
            net.dense.weight.zero_()
            net.dense.bias.fill_(-3.0)
            net.dense.weight[:n_out] = old.weight
            net.dense.bias[:n_out] = old.bias
    return net.to(dev), n_out


def medir(lv, yv):
    aps, aus = [], []
    for k in range(7):
        m = yv[:, k] >= 0
        ok = m.sum() > 0 and (yv[m, k] == 1).any() and (yv[m, k] == 0).any()
        aps.append(R.avg_prec(yv[m, k], lv[m, k]) if ok else float("nan"))
        aus.append(tt.auroc(yv[m, k], lv[m, k]) if ok else float("nan"))
    ok = ~np.isnan(aps)
    return float((np.array(aps)[ok] * PESO[ok]).sum() / PESO[ok].sum()), aps, aus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t202")
    ap.add_argument("--init", default="/root/holter/t200/runs/noche/best.pt")
    ap.add_argument("--tags", default="holter,icentia,taqui")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_bb", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_cap", type=int, default=30000)
    ap.add_argument("--paciencia", type=int, default=4)
    ap.add_argument("--val_tags", default="holter,icentia,taqui")
    ap.add_argument("--rob", type=float, default=0.0, help="prob. de perturbar canales en train")
    ap.add_argument("--s_neg_w", type=float, default=1.0, help="peso de la perdida de S en ventanas sin S (menos S falsas)")
    ap.add_argument("--s_negv_w", type=float, default=1.0, help="peso extra si ademas la ventana tiene V")
    ap.add_argument("--s_samp", type=float, default=1.0, help="muestreo extra de ventanas con V y sin S")
    args = ap.parse_args()
    torch.manual_seed(202)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(202)
    dev = torch.device("cuda:0")
    out = Path("/root/holter/t200/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    R.banco()
    srcs = load_sources7(args.tags.split(","))
    tr = T.pairs_for(srcs, "train")
    srcs_v = load_sources7(args.val_tags.split(","))
    va_all = T.pairs_for(srcs_v, "val", regular_only=True)
    yva = np.stack([y7(srcs_v[s], j) for s, j in va_all])
    raro = (yva[:, 5] == 1) | (yva[:, 6] == 1)
    keep = np.where(raro)[0]
    rest = np.where(~raro)[0]
    n_rest = max(args.val_cap - len(keep), 0)
    if len(rest) > n_rest:
        rest = rng.choice(rest, n_rest, replace=False)
    va = [va_all[i] for i in sorted(np.concatenate([keep, rest]))]
    dbs_tr = np.array([srcs[s].db[j] for s, j in tr])
    names, counts = np.unique(dbs_tr, return_counts=True)
    w_db = {n: 1.0 / math.sqrt(c) for n, c in zip(names, counts, strict=True)}
    ytr = np.stack([y7(srcs[s], j) for s, j in tr])
    weights = np.array([w_db[d] for d in dbs_tr])
    weights = weights * np.where((ytr[:, 5] == 1) | (ytr[:, 6] == 1), 8.0, 1.0)
    weights = weights * np.where((ytr[:, 2] == 1) & (ytr[:, 3] == 0), args.s_samp, 1.0)
    pos = ((ytr == 1) * weights[:, None]).sum(0)
    neg = ((ytr == 0) * weights[:, None]).sum(0)
    pw = np.clip(np.sqrt(np.maximum(neg, 1e-9) / np.maximum(pos, 1e-9)), 1.0, 30.0)
    pw[3], pw[4] = 1.0, 2.0
    pw[5:] = np.clip(pw[5:], 1.0, 3.0)
    pw_t = torch.tensor(pw, dtype=torch.float32, device=dev)
    log(f"train={len(tr)} val={len(va)} bases={dict(zip(names.tolist(), counts.tolist(), strict=True))}")
    log(f"positivos train {dict(zip(TARGETS7, (ytr == 1).sum(0).astype(int).tolist(), strict=True))} "
        f"val {dict(zip(TARGETS7, (np.stack([y7(srcs_v[s], j) for s, j in va]) == 1).sum(0).astype(int).tolist(), strict=True))}")
    log(f"pos_weight={np.round(pw, 2).tolist()}")
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=10**8, replacement=True)
    dl = DataLoader(DS7(srcs, tr, "train", rob=args.rob), batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, prefetch_factor=4)
    dl_va = DataLoader(DS7(srcs_v, va, "val", seed=202), batch_size=256, num_workers=args.workers, pin_memory=True)
    dl_vr = DataLoader(DS7(srcs_v, va, "valrob", seed=202), batch_size=256, num_workers=args.workers, pin_memory=True)

    net, n0 = make_model7(dev, args.init)
    log(f"modelo inicial {args.init} con {n0} salidas -> 7")
    head = list(net.dense.parameters())
    bb = [p for n, p in net.named_parameters() if not n.startswith("dense.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_bb}, {"params": head, "lr": args.lr_head}], weight_decay=args.wd)
    base_lr = [args.lr_bb, args.lr_head]
    scaler = torch.amp.GradScaler("cuda")
    hist, best, sin_mejora, step = [], -1.0, 0, 0

    def evaluar(el):
        nonlocal best, sin_mejora
        lv, yv = T.predict(net, dl_va, dev)
        mv, aps, aus = medir(lv, yv)
        m3 = yv[:, 3] >= 0
        pp = R.ppv_at_se(yv[m3, 3], lv[m3, 3], 0.85)
        hist.append({"min": round(el / 60, 1), "step": step, "puntaje_ap": mv, "ap": aps, "auroc": aus, "S_vpp_a_se085": pp})
        log(f"VAL min={el / 60:.1f} puntaje_AP={mv:.4f} AP={[round(x, 4) for x in aps]} "
            f"AUROC={[round(x, 4) for x in aus]} S_VPP@Se0.85={pp:.3f}")
        if args.rob:
            lr_, yr_ = T.predict(net, dl_vr, dev)
            mr, apr, _ = medir(lr_, yr_)
            m3r = yr_[:, 3] >= 0
            log(f"VALROB (canales reordenados/invertidos) puntaje_AP={mr:.4f} AP={[round(x, 4) for x in apr]} "
                f"S_VPP@Se0.85={R.ppv_at_se(yr_[m3r, 3], lr_[m3r, 3], 0.85):.3f} | normal={mv:.4f}")
            hist[-1]["puntaje_normal"], hist[-1]["puntaje_rob"], hist[-1]["ap_rob"] = mv, mr, apr
            mv = (mv + mr) / 2
        ck = {"state_dict": net.state_dict(), "step": step, "puntaje_ap": mv, "targets": TARGETS7, "base": str(args.init)}
        torch.save(ck, out / "last.pt")
        if el > 0:
            if mv > best + 1e-4:
                best, sin_mejora = mv, 0
                torch.save(ck, out / "best.pt")
                log("  nuevo mejor -> best.pt")
            else:
                sin_mejora += 1
                log(f"  sin mejora ({sin_mejora}/{args.paciencia})")
        else:  # el punto de partida cuenta como mejor: una fase nunca guarda algo peor que su inicio
            best = mv
            torch.save(ck, out / "best.pt")
        (out / "hist.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")

    evaluar(0)
    budget = args.minutes * 60
    t_start = time.time()
    next_eval = args.every * 60
    run_loss, nb = 0.0, 0
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
        elem = nn.functional.binary_cross_entropy_with_logits(logits, yb.clamp(min=0), pos_weight=pw_t, reduction="none")
        if args.s_neg_w != 1.0 or args.s_negv_w != 1.0:
            w3 = torch.where(yb[:, 3] == 0, torch.where(yb[:, 2] == 1, args.s_neg_w * args.s_negv_w, args.s_neg_w), 1.0)
            elem = torch.cat([elem[:, :3], elem[:, 3:4] * w3[:, None], elem[:, 4:]], 1)
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
            log(f"paso {step} min={el / 60:.1f} loss={run_loss / nb:.4f} {step * args.bs / el:.0f} ventanas/s lr_mult={mult:.3f}")
            run_loss, nb = 0.0, 0
        if el >= next_eval:
            next_eval += args.every * 60
            evaluar(el)
            if sin_mejora >= args.paciencia:
                log("PARADA: ya no mejora")
                break
            net.train()
    else:
        pass
    if sin_mejora < args.paciencia:
        evaluar(time.time() - t_start)
    log(f"FIN pasos={step} mejor={best:.4f}")
    os._exit(0)


if __name__ == "__main__":
    main()
