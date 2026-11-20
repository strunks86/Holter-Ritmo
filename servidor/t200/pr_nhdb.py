"""PR largo en Holter real: NHDB. Positivo = segmento con evento AVB (grado no indicado), sin FA/flutter. Negativo = sin eventos.
No se usa NHDB para entrenar PR, asi que todas las ventanas sirven. Ventanas con ruido ilegible fuera."""
import sys, glob, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S
T, tt = S.T, S.tt
dev = torch.device("cuda:0")
D = glob.glob("/root/holter/data/HolterIA/bases2/noise-holter-database/noise*/wfdb_files")[0]
ev = {}
for h in glob.glob(D + "/*.hea"):
    com = [x for x in open(h).read().split("\n") if x.startswith("#")]
    parts = [p.strip() for p in com[0][1:].split("|")] if com else []
    i = next((k for k, p in enumerate(parts) if p.startswith("EVENTS")), None)
    ev[os.path.basename(h)[:-4]] = set([parts[i].split(":", 1)[1].strip()] + parts[i + 1:]) if i is not None else set()
X = np.load(T.NEW / "n_nhdb_X.npy", mmap_mode="r"); m = np.load(T.NEW / "n_nhdb_meta.npz"); TQ = np.load(T.NEW / "n_nhdb_tq.npz")["TQ"]
y = np.full(len(X), -1.0)
for i, r in enumerate(m["rec"]):
    e = ev.get(str(r), set()) - {""}
    if TQ[i, 3] == 1: continue
    if "AVB" in e and not e & {"A. Fib.", "A. Flutter"}: y[i] = 1
    elif e <= {"None"}: y[i] = 0
ii = np.where((y >= 0) & ((m["split"] == "test") if "--test" in sys.argv else True))[0]; print("ventanas AVB", int((y == 1).sum()), "sin eventos", int((y == 0).sum()), "registros AVB", len({m["rec"][k] for k in ii if y[k] == 1}), flush=True)
for mp in sys.argv[1].split(","):
    net, _ = S.make_model7(dev, Path("/root/holter/t200/runs/t205b/sopa50.pt")); net.dense = nn.Linear(1024, 1).to(dev)
    net.load_state_dict(torch.load(mp, weights_only=False)["state_dict"]); net.eval(); L = []
    with torch.no_grad():
        for k in range(0, len(ii), 256):
            xb = torch.from_numpy(np.stack([np.asarray(X[j], dtype=np.float32).T for j in ii[k:k + 256]])).to(dev)
            with torch.autocast("cuda", dtype=torch.float16): L.append(net(tt.to12(xb)).float().cpu().numpy()[:, 0])
    l = np.concatenate(L); yy = y[ii]; p = 1 / (1 + np.exp(-l)); o = np.argsort(-l); t = yy[o] == 1; tp = np.cumsum(t); ppv = tp / np.arange(1, len(o) + 1)
    print(mp, "AUROC", round(float(tt.auroc(yy, l)), 3), "AP", round(float(ppv[t].mean()), 3), "prevalencia", round(float(yy.mean()), 3),
          "Se@0.5", round(float((p[yy == 1] > 0.5).mean()), 3), "FP/1000@0.5", round(float(1000 * (p[yy == 0] > 0.5).mean()), 1), flush=True)
    import collections; suj = np.array([str(m["rec"][k])[:3] for k in ii]); fil = collections.defaultdict(list)
    for sj, pp, yv in zip(suj, p, yy): fil[(sj, int(yv))].append(pp)
    for (sj, yv), v in sorted(fil.items(), key=lambda kv: -np.mean(kv[1]))[:14]:
        print(f"   sujeto {sj} {'AVB' if yv else 'sin eventos'}: ventanas {len(v)}, marcadas {np.mean(np.array(v) > 0.5):.2f}, p media {np.mean(v):.2f}", flush=True)
print("TERMINADO"); os._exit(0)
