import glob, os, collections, numpy as np
D = glob.glob("/root/holter/data/HolterIA/bases2/noise-holter-database/noise*/wfdb_files")[0]
s = collections.defaultdict(lambda: [0, 0])
for h in glob.glob(D + "/*.hea"):
    t = open(h).read(); k = os.path.basename(h)[:3]; s[k][1] += 1; s[k][0] += "AVB" in t
m = np.load("/root/holter/t200/data/n_nhdb_meta.npz"); sp = {r[:3]: x for r, x in zip(m["rec"], m["split"])}
con = [(k, v[0], v[1], sp.get(k)) for k, v in sorted(s.items()) if v[0]]
print("sujetos con AVB", len(con), "de", len(s)); print(con)
for x in ("train", "val", "test"): print(x, "sujetos", sum(1 for k in s if sp.get(k) == x), "con AVB", sum(1 for c in con if c[3] == x))
