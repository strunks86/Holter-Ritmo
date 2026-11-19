"""Diagnostico: T204 vs T205 por base nueva (split val), AUROC/AP por salida, y cruces FA/FLUT."""
import sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S
T, tt = S.T, S.tt
N = ["FA", "FLUT", "PVC", "PAC", "RUIDO", "TV", "TSV"]

class D(Dataset):
    def __init__(s, src, idx): s.src, s.idx = src, idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = int(s.idx[i]); y = S.y7(s.src, j); y0 = y.copy(); y0[0] = s.src.Y[j, 0]
        return torch.from_numpy(np.asarray(s.src.X[j], dtype=np.float32).T.copy()), torch.from_numpy(y0)

def ap(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); return float((tp / np.arange(1, len(y) + 1))[y == 1].mean())

