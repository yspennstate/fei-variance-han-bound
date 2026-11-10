"""Checks for the one-coordinate lemma and the adaptive theorem.

For random f: {-1,1}^n -> {-1,1} and each coordinate i:
  D_i = H(f) - H(f_+)/2 - H(f_-)/2, computed from three separate Walsh-Hadamard transforms;
  E_i = H(1[i in U] | U minus i), computed from the spectrum of f by grouping S and S+i;
  J_i = Jensen-Shannon divergence of the spectral laws of f_+ and f_-.
Checks D_i = E_i + J_i, J_i <= (2/ln 2) Inf_i, E_i <= (2 + 1/ln 2) d + d log2(1 + Var/d).
Then the right side of the adaptive theorem along random adaptive trees (the next coordinate is drawn at
random among the relevant ones at every node), compared with H(f) and with the fixed-order right side."""
import math
import numpy as np

LN2 = math.log(2)
C0 = 2 + 3 / LN2


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


def ent(p):
    p = p[p > 1e-300]
    return float(-(p * np.log2(p)).sum())


def restrict(f, n, i, b):
    """f with x_i fixed; bit 1 of index <-> x = -1. Returns a table on n-1 variables."""
    idx = np.arange(len(f))
    keep = idx[((idx >> i) & 1) == b]
    return f[keep]


def infl(f, n, i):
    idx = np.arange(len(f))
    return float(np.mean(f != f[idx ^ (1 << i)]))


def var(f):
    m = float(np.mean(f))
    return 1.0 - m * m


def step_terms(f, n, i):
    c = wht(f)
    p = c * c
    H = ent(p)
    fp, fm = restrict(f, n, i, 0), restrict(f, n, i, 1)
    pp, pm = wht(fp) ** 2, wht(fm) ** 2
    D = H - 0.5 * ent(pp) - 0.5 * ent(pm)
    idx = np.arange(len(f))
    lo = idx[((idx >> i) & 1) == 0]
    a, b = p[lo], p[lo | (1 << i)]
    s = a + b
    m = s > 1e-300
    E = float(sum(s[m] * np.array([0 if x <= 0 or x >= 1 else -(x * math.log2(x) + (1 - x) * math.log2(1 - x))
                                    for x in (b[m] / s[m])])))
    mix = 0.5 * (pp + pm)
    J = ent(mix) - 0.5 * ent(pp) - 0.5 * ent(pm)
    return H, D, E, J


def adaptive_rhs(f, n, rng):
    """c0*Inf + Inf*log2(1+Var/Inf) summed along one random adaptive tree, averaged exactly over the inputs."""
    if np.all(f == f[0]):
        return 0.0
    rel = [i for i in range(n) if infl(f, n, i) > 0]
    i = int(rng.choice(rel))
    d = infl(f, n, i)
    v = var(f)
    here = C0 * d + d * math.log2(1 + v / d)
    return here + 0.5 * adaptive_rhs(restrict(f, n, i, 0), n - 1, rng) + 0.5 * adaptive_rhs(restrict(f, n, i, 1), n - 1, rng)


def fixed_rhs(f, n):
    v = var(f)
    tot = 0.0
    for k in range(n):
        d = infl(f, n, k)
        if d > 0:
            tot += C0 * d + d * math.log2(1 + v / d)
    return tot


rng = np.random.default_rng(7)
worst_id = worst_J = worst_E = worst_D = 0.0
nsteps = 0
for n in range(2, 9):
    N = 1 << n
    for bias in (0.5, 0.2, 0.05, 0.02):
        for _ in range(40):
            f = np.where(rng.random(N) < bias, -1, 1)
            if np.all(f == f[0]):
                continue
            V = var(f)
            for i in range(n):
                d = infl(f, n, i)
                H, D, E, J = step_terms(f, n, i)
                worst_id = max(worst_id, abs(D - E - J))
                nsteps += 1
                if d == 0:
                    assert abs(D) < 1e-9 and abs(E) < 1e-9 and abs(J) < 1e-9
                    continue
                worst_J = max(worst_J, J / (2 / LN2 * d))
                worst_E = max(worst_E, E / ((2 + 1 / LN2) * d + d * math.log2(1 + V / d)))
                worst_D = max(worst_D, D / (C0 * d + d * math.log2(1 + V / d)))
print(f"one-coordinate steps checked: {nsteps}")
print(f"max |D - E - J| = {worst_id:.2e}")
print(f"max J / ((2/ln2) Inf) = {worst_J:.4f}")
print(f"max E / ((2+1/ln2) d + d log2(1+Var/d)) = {worst_E:.4f}")
print(f"max D / (c0 d + d log2(1+Var/d)) = {worst_D:.4f}")

worst_a = 0.0
worse_than_fixed = 0
trees = 0
for n in range(2, 8):
    N = 1 << n
    for bias in (0.5, 0.2, 0.05):
        for _ in range(15):
            f = np.where(rng.random(N) < bias, -1, 1)
            if np.all(f == f[0]):
                continue
            H = ent(wht(f) ** 2)
            fx = fixed_rhs(f, n)
            for _t in range(4):
                a = adaptive_rhs(f, n, rng)
                trees += 1
                worst_a = max(worst_a, H / a)
                if a > fx + 1e-12:
                    worse_than_fixed += 1
print(f"random adaptive trees: {trees}, max H / adaptive right side = {worst_a:.4f}")
print(f"trees whose adaptive right side exceeds the fixed-order one: {worse_than_fixed} (allowed; Jensen runs one way)")
