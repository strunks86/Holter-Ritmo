"""Promedio de pesos de dos modelos (misma arquitectura): sopa.py a.pt b.pt peso_a salida.pt"""
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
w = float(sys.argv[3]); sd = {}
for k, v in a["state_dict"].items():
    u = b["state_dict"][k]
    sd[k] = (w * v + (1 - w) * u) if v.is_floating_point() else v
a["state_dict"] = sd; a["base"] = f"sopa {w} {sys.argv[1]} + {sys.argv[2]}"
torch.save(a, sys.argv[4]); print("ok", sys.argv[4])
