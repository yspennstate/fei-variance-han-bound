"""Check H(f) <= (2 + 3/ln 2) I(f) + sum_k I_k log2(1 + Var(f)/I_k) on random Boolean functions,
and compare with Han's right side (3/ln 2) I + sum_k I_k log2(4/I_k).
f takes values in {-1, 1}; Var(f) = 1 - fhat(empty)^2; I_k = P[f(x) != f(x^k)]."""
import math
import numpy as np


def wht(v):
    a = v.astype(float).copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        x, y = a[:, :h].copy(), a[:, h:].copy()
        a[:, :h], a[:, h:] = x + y, x - y
        a = a.reshape(-1)
        h *= 2
    return a / len(a)


def stats(f, n):
    c = wht(f)
    p = c * c
    nz = p[p > 1e-300]
    H = float(-(nz * np.log2(nz)).sum())
    idx = np.arange(len(f))
    Ik = []
    for k in range(n):
        Ik.append(float(np.mean(f != f[idx ^ (1 << k)])))
    var = 1.0 - c[0] ** 2
    return H, Ik, var


def rhs_var(Ik, var):
    C = 2 + 3 / math.log(2)
    return C * sum(Ik) + sum(t * math.log2(1 + var / t) for t in Ik if t > 0)


def rhs_han(Ik):
    return 3 / math.log(2) * sum(Ik) + sum(t * math.log2(4 / t) for t in Ik if t > 0)


rng = np.random.default_rng(1)
worst = 0.0
rows = 0
for n in range(2, 11):
    N = 1 << n
    for bias in (0.5, 0.2, 0.05, 0.01):
        for _ in range(60):
            f = np.where(rng.random(N) < bias, -1, 1)
            if abs(f.sum()) == N:
                continue
            H, Ik, var = stats(f, n)
            worst = max(worst, H / rhs_var(Ik, var))
            rows += 1
# AND_k: separation from Han
print("random functions:", rows, "max H / var-bound =", round(worst, 4))
print(" k   H/I    var-bound/I   Han/I")
for k in range(2, 15):
    N = 1 << k
    f = np.ones(N, dtype=int)
    f[N - 1] = -1
    H, Ik, var = stats(f, k)
    I = sum(Ik)
    print(f"{k:2d} {H / I:6.3f} {rhs_var(Ik, var) / I:10.3f} {rhs_han(Ik) / I:9.3f}")
