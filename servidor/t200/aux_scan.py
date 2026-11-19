import wfdb, glob, collections, os, sys
B="/root/holter/data/HolterIA/bases"
for db,pat in [("mitdb","*.atr"),("svdb","*.atr"),("incartdb","*.atr"),("ltdb","*.atr"),("edb","*.atr"),("sddb","*.atr"),("shdb-af","*.atr"),("ltafdb","*.atr"),("afdb","*.atr"),("cpsc2021","*/*.atr")]:
    c=collections.Counter(); sy=collections.Counter()
    for p in sorted(glob.glob(f"{B}/{db}/{pat}")):
        try: a=wfdb.rdann(p[:-4],"atr")
        except Exception as e: continue
        c.update([x.strip("\x00 ") for x in a.aux_note if x and x.strip("\x00 ").startswith("(")]); sy.update(a.symbol)
    print(db, dict(c.most_common(25)), "| simbolos", dict(sy.most_common(14)), flush=True)
import csv, ast
c=collections.Counter()
for r in csv.DictReader(open(f"{B}/ptb-xl/ptbxl_database.csv",encoding="utf-8",errors="replace")):
    c.update(ast.literal_eval(r["scp_codes"]).keys())
print("ptbxl", {k:c[k] for k in ["SVTAC","PSVT","SVARR","AFIB","AFLT","STACH","SR","BIGU","TRIGU","PVC","PAC","VT"] if k in c})
ps=sorted(glob.glob(f"{B}/icentia11k/p00/p0000*/*.atr"))[:40]
c=collections.Counter()
for p in ps:
    a=wfdb.rdann(p[:-4],"atr"); c.update([x for x in a.aux_note if x and x!="None"])
print("icentia", dict(c))
