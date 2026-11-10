"""merge_census.py - merge the part files written by mono_census and recompute the leaders independently.

usage: python merge_census.py OUT_DIR PREFIX [TOP]      e.g.  python merge_census.py ../results out7_ 5

Reads OUT_DIR/PREFIX*.txt, checks that the weighted counts add up to the Dedekind number, merges the top
lists per statistic, and recomputes each printed function from its truth table with a separate transform
(plain floating-point Walsh-Hadamard in numpy, logs taken directly, no integer table).
"""
import glob
import math
import os
import sys

import numpy as np

DEDEKIND = {5: 7581, 6: 7828354, 7: 2414682040998}


def wht(v):
    v = v.astype(np.float64).copy()
    h = 1
    n = len(v)
    while h < n:
        v = v.reshape(-1, 2, h)
        a = v[:, 0, :] + v[:, 1, :]
        b = v[:, 0, :] - v[:, 1, :]
        v = np.stack([a, b], axis=1).reshape(-1)
        h *= 2
    return v / n


def eta(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = -x[m] * np.log2(x[m])
    return out


def stats(f1, f0, K):
    """f0 = restriction x_n = 0 (low half), f1 = restriction x_n = 1; bit x of the table set <=> value -1."""
    n = K + 1
    half = 1 << K
    N = 1 << n
    vals = np.empty(N)
    for x in range(half):
        vals[x] = 1 - 2 * ((f0 >> x) & 1)
        vals[x + half] = 1 - 2 * ((f1 >> x) & 1)
    fh = wht(vals)
    mu = fh * fh
    H = float(eta(mu).sum())
    pop = np.array([bin(t).count("1") for t in range(N)])
    I = float((pop * mu).sum())
    infl, E, D = [], [], []
    idx = np.arange(N)
    for i in range(n):
        bit = 1 << i
        T = idx[(idx & bit) == 0]
        a = fh[T]
        b = fh[T | bit]
        inf = float((b * b).sum())
        infl.append(inf)
        if inf == 0:
            E.append(float("nan"))
            D.append(float("nan"))
            continue
        E.append(H - float(eta(a * a + b * b).sum()))
        D.append(H - 0.5 * float((eta((a + b) ** 2) + eta((a - b) ** 2)).sum()))
    out = {"H": H, "I": I, "mean": float(fh[0]), "inf": infl, "E": E, "D": D}
    pos = [i for i in range(n) if infl[i] > 0]
    out["HI"] = H / I
    out["minE"] = min(E[i] / infl[i] for i in pos)
    out["minD"] = min(D[i] / infl[i] for i in pos)
    mx = max(infl[i] for i in pos)
    mn = min(infl[i] for i in pos)
    out["ruleMax"] = min(E[i] / infl[i] for i in pos if abs(infl[i] - mx) < 1e-12)
    out["ruleMin"] = min(E[i] / infl[i] for i in pos if abs(infl[i] - mn) < 1e-12)
    if abs(fh[0]) < 1e-12 and I > 1 + 1e-12:
        out["balHI1"] = H / (I - 1)
    for b, name in ((0.0, "Q0"), (0.5, "Q05"), (1.0, "Q1")):
        g = [(1 + math.log2(1 / infl[i])) ** (-b) for i in pos]
        out[name] = sum(g[k] * E[i] for k, i in enumerate(pos)) / sum(g[k] * infl[i] for k, i in enumerate(pos))
    out["Q1mQ0"] = out["Q1"] - out["Q0"]
    var = 1 - float(mu[0])
    svar = sum(infl[i] * math.log2(1 + var / infl[i]) for i in pos)
    shan = sum(infl[i] * math.log2(4 / infl[i]) for i in pos)
    out["HBvar"] = H / ((2 + 3 / math.log(2)) * I + svar)
    out["HBhan"] = H / ((3 / math.log(2)) * I + shan)
    return out


def main():
    d, prefix = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    files = sorted(glob.glob(os.path.join(d, prefix + "*.txt")))
    tot_w = tot_e = 0
    recs = {}
    K = None
    parts = []
    for fn in files:
        with open(fn, encoding="utf-8") as fh:
            head = fh.readline().split()
            kv = dict(t.split("=") for t in head[1:])
            K = int(kv["K"])
            parts.append((int(kv["p"]), int(kv["P"])))
            tot_w += int(kv["weighted_count"])
            tot_e += int(kv["evals"])
            for line in fh:
                s, v, f1, f0, H, I = line.split()
                recs.setdefault(s, []).append((float(v), int(f1, 16), int(f0, 16), float(H), float(I)))
    n = K + 1
    P = parts[0][1]
    print("files", len(files), "parts", sorted(p for p, _ in parts), "of", P, " n =", n)
    print("weighted_count sum", tot_w, " Dedekind", DEDEKIND.get(n), " match", tot_w == DEDEKIND.get(n), " evals", tot_e)
    for s in recs:
        lst = sorted(set(recs[s]), reverse=True)
        print("==", s, "entries", len(lst))
        for v, f1, f0, H, I in lst[:top]:
            st = stats(f1, f0, K)
            mine = st.get(s, float("nan"))
            print("  %.12f  f1=%0*x f0=%0*x  H=%.9f I=%.9f  recomputed=%.12f diff=%.1e  mean=%.6f"
                  % (v, (1 << K) // 4, f1, (1 << K) // 4, f0, H, I, mine, abs(mine - v), st["mean"]))
            print("      inf", " ".join("%.6f" % x for x in st["inf"]))
            print("      E/inf", " ".join("%.6f" % (st["E"][i] / st["inf"][i]) if st["inf"][i] > 0 else "nan" for i in range(n)))


if __name__ == "__main__":
    main()
