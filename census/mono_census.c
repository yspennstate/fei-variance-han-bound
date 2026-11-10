/* mono_census.c - exhaustive census of monotone Boolean functions on n = K+1 variables (K <= 6).
   A function is a pair (f0, f1) of monotone K-variable functions with f0 <= f1 (f0 = restriction x_n = 0).
   f0 runs over one representative of each orbit of S_K acting on the first K variables; f1 over all
   monotone functions above it.  Every statistic below is invariant under permuting variables, so every
   n-variable monotone function is covered up to symmetry.  Check: sum over representatives of
   orbit_size * #{f1 >= f0} must be the Dedekind number (7,828,354 for n = 6; 2,414,682,040,998 for n = 7).

   Per function (Fourier coefficients fhat(T) = F[T]/N, N = 2^n, mu = fhat^2):
     H = sum_T eta(mu(T)),  I = sum_T |T| mu(T),  eta(x) = x log2(1/x)
     for each coordinate i with a = fhat(T), b = fhat(T+i), T not containing i:
       Inf_i = sum b^2
       E_i = sum [eta(a^2) + eta(b^2) - eta(a^2+b^2)]            (erasure entropy H(1[i in U] | U - i))
       D_i = sum [eta(a^2) + eta(b^2) - eta((a+b)^2)/2 - eta((a-b)^2)/2]   (one-step entropy drop)
   Statistics kept (top lists): H/I; minE = min_i E_i/Inf_i; minD = min_i D_i/Inf_i; ruleMax = E/Inf at
   the coordinate of largest influence (best among ties); ruleMin = same at the smallest positive
   influence; balHI1 = H/(I-1) for balanced functions with I > 1.

   usage: mono_census K P p outfile     (process p of P handles representatives with index % P == p)  */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef unsigned long long u64;

static u64 *M[7];
static size_t cnt[7];
static int K, n, N, half;
static double *eta;            /* eta[m] = (m/N^2) log2(N^2/m), m = 0 .. 4 N^2 */

/* Q_b = sum_i g(Inf_i) E_i / sum_i g(Inf_i) Inf_i, g(t) = (1 + log2(1/t))^-b, b = 0, 1/2, 1;
   Q1mQ0 = Q_1 - Q_0. */
/* HBvar = H / B_var, B_var = (2 + 3/ln 2) I + sum_k Inf_k log2(1 + Var/Inf_k), Var = 1 - fhat(0)^2;
   HBhan = H / B_han, B_han = (3/ln 2) I + sum_k Inf_k log2(4/Inf_k). */
#define NSTAT 12
#define TOPK 300
static const char *statname[NSTAT] = {"HI", "minE", "minD", "ruleMax", "ruleMin", "balHI1", "Q0", "Q05", "Q1", "Q1mQ0", "HBvar", "HBhan"};
typedef struct { double v; u64 f0, f1; double H, I; } rec;
static rec top[NSTAT][TOPK];
static int ntop[NSTAT];
static double thr[NSTAT];

static void push(int s, double v, u64 f0, u64 f1, double H, double I) {
    if (ntop[s] < TOPK) {
        top[s][ntop[s]++] = (rec){v, f0, f1, H, I};
        if (ntop[s] == TOPK) { double m = 1e300; for (int k = 0; k < TOPK; k++) if (top[s][k].v < m) m = top[s][k].v; thr[s] = m; }
        return;
    }
    if (v <= thr[s]) return;
    int arg = 0; double m = 1e300;
    for (int k = 0; k < TOPK; k++) if (top[s][k].v < m) { m = top[s][k].v; arg = k; }
    top[s][arg] = (rec){v, f0, f1, H, I};
    m = 1e300; for (int k = 0; k < TOPK; k++) if (top[s][k].v < m) m = top[s][k].v;
    thr[s] = m;
}

static void build(void) {
    M[0] = malloc(2 * sizeof(u64)); M[0][0] = 0; M[0][1] = 1; cnt[0] = 2;
    for (int k = 0; k < K; k++) {
        size_t cap = 1024, c = 0; u64 *out = malloc(cap * sizeof(u64));
        int sh = 1 << k;
        for (size_t j1 = 0; j1 < cnt[k]; j1++) for (size_t j0 = 0; j0 < cnt[k]; j0++) {
            u64 g0 = M[k][j0], g1 = M[k][j1];
            if (g0 & ~g1) continue;
            if (c == cap) { cap *= 2; out = realloc(out, cap * sizeof(u64)); }
            out[c++] = g0 | (g1 << sh);
        }
        M[k + 1] = out; cnt[k + 1] = c;
    }
}

static long find(u64 f) {           /* index of f in the sorted list M[K] */
    long lo = 0, hi = (long)cnt[K] - 1;
    while (lo <= hi) { long mid = (lo + hi) / 2; if (M[K][mid] == f) return mid; if (M[K][mid] < f) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

static int nperm; static unsigned char (*ptab)[64];
static void gen_perms(void) {
    int p[6], used[6] = {0}; int fact = 1; for (int i = 2; i <= K; i++) fact *= i;
    ptab = malloc(fact * 64); nperm = 0;
    int idx[6] = {0};
    /* iterative permutation generation by counting in factorial base */
    for (int code = 0; code < fact; code++) {
        int c = code, avail[6], na = K; for (int i = 0; i < K; i++) avail[i] = i;
        for (int i = 0; i < K; i++) { int base = na; int d = c % base; c /= base; p[i] = avail[d]; for (int j = d; j < na - 1; j++) avail[j] = avail[j + 1]; na--; }
        for (int x = 0; x < (1 << K); x++) { int y = 0; for (int i = 0; i < K; i++) if (x >> i & 1) y |= 1 << p[i]; ptab[nperm][x] = (unsigned char)y; }
        nperm++;
    }
    (void)used; (void)idx;
}
static u64 apply(u64 f, int q) { u64 g = 0; int sz = 1 << K; for (int x = 0; x < sz; x++) if (f >> ptab[q][x] & 1) g |= 1ULL << x; return g; }

static u64 evals = 0;
static void evaluate(u64 f0, u64 f1) {
    short F[128];
    for (int x = 0; x < half; x++) { F[x] = 1 - 2 * (int)(f0 >> x & 1); F[x + half] = 1 - 2 * (int)(f1 >> x & 1); }
    for (int len = 1; len < N; len <<= 1)
        for (int i = 0; i < N; i += 2 * len)
            for (int j = i; j < i + len; j++) { short u = F[j], v = F[j + len]; F[j] = u + v; F[j + len] = u - v; }
    evals++;
    int sq[128]; double H = 0; long Inum = 0;
    for (int T = 0; T < N; T++) { sq[T] = (int)F[T] * F[T]; H += eta[sq[T]]; Inum += (long)__builtin_popcount(T) * sq[T]; }
    if (Inum == 0) return;
    double NN = (double)N * N, I = Inum / NN;
    double minE = 1e300, minD = 1e300, rmaxv = 1e300, rminv = 1e300; long infmax = -1, infmin = 1L << 60;
    double qn0 = 0, qd0 = 0, qn05 = 0, qd05 = 0, qn1 = 0, qd1 = 0;
    double Var = 1.0 - sq[0] / NN, svar = 0, shan = 0;
    for (int i = 0; i < n; i++) {
        int bit = 1 << i; long infn = 0; double S = 0, X = 0;
        for (int T = 0; T < N; T++) {
            if (T & bit) continue;
            int A = F[T], B = F[T | bit];
            infn += sq[T | bit];
            S += eta[sq[T] + sq[T | bit]];
            X += eta[(A + B) * (A + B)] + eta[(A - B) * (A - B)];
        }
        if (infn == 0) continue;
        double inf = infn / NN, E = H - S, D = H - 0.5 * X;
        double re = E / inf, rd = D / inf;
        double g1 = 1.0 / (1.0 + log2(1.0 / inf)), g05 = sqrt(g1);
        qn0 += E; qd0 += inf; qn05 += g05 * E; qd05 += g05 * inf; qn1 += g1 * E; qd1 += g1 * inf;
        svar += inf * log2(1.0 + Var / inf); shan += inf * log2(4.0 / inf);
        if (re < minE) minE = re;
        if (rd < minD) minD = rd;
        if (infn > infmax) { infmax = infn; rmaxv = re; } else if (infn == infmax && re < rmaxv) rmaxv = re;
        if (infn < infmin) { infmin = infn; rminv = re; } else if (infn == infmin && re < rminv) rminv = re;
    }
    push(0, H / I, f0, f1, H, I);
    push(1, minE, f0, f1, H, I);
    push(2, minD, f0, f1, H, I);
    push(3, rmaxv, f0, f1, H, I);
    push(4, rminv, f0, f1, H, I);
    if (F[0] == 0 && Inum > (long)NN) push(5, H / (I - 1.0), f0, f1, H, I);
    push(6, qn0 / qd0, f0, f1, H, I);
    push(7, qn05 / qd05, f0, f1, H, I);
    push(8, qn1 / qd1, f0, f1, H, I);
    push(9, qn1 / qd1 - qn0 / qd0, f0, f1, H, I);
    push(10, H / ((2.0 + 3.0 / M_LN2) * I + svar), f0, f1, H, I);
    push(11, H / ((3.0 / M_LN2) * I + shan), f0, f1, H, I);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: mono_census K P p outfile\n"); return 2; }
    K = atoi(argv[1]); int P = atoi(argv[2]), p = atoi(argv[3]);
    if (K < 1 || K > 6) return 2;
    n = K + 1; N = 1 << n; half = N / 2;
    long tabsz = 4L * N * N + 1; eta = malloc(tabsz * sizeof(double)); eta[0] = 0;
    for (long m = 1; m < tabsz; m++) { double x = (double)m / ((double)N * N); eta[m] = -x * log2(x); }
    for (int s = 0; s < NSTAT; s++) thr[s] = -1e300;
    build(); gen_perms();
    size_t c = cnt[K];
    unsigned char *vis = calloc(c, 1);
    u64 weighted = 0, nrep = 0, mine = 0;
    for (size_t idx = 0; idx < c; idx++) {
        if (vis[idx]) continue;
        u64 f0 = M[K][idx];
        u64 orbit = 0;
        for (int q = 0; q < nperm; q++) { long j = find(apply(f0, q)); if (j < 0) { fprintf(stderr, "orbit image not monotone\n"); return 3; } if (!vis[j]) { vis[j] = 1; orbit++; } }
        u64 r = nrep++;
        if ((int)(r % P) != p) continue;
        mine++;
        u64 above = 0;
        for (size_t j = 0; j < c; j++) { u64 f1 = M[K][j]; if (f0 & ~f1) continue; above++; evaluate(f0, f1); }
        weighted += orbit * above;
    }
    FILE *o = fopen(argv[4], "w");
    fprintf(o, "# K=%d n=%d P=%d p=%d monotone_K=%zu reps_total=%llu reps_mine=%llu evals=%llu weighted_count=%llu\n",
            K, n, P, p, c, nrep, mine, evals, weighted);
    for (int s = 0; s < NSTAT; s++) for (int k = 0; k < ntop[s]; k++)
        fprintf(o, "%s %.15f %016llx %016llx %.15f %.15f\n", statname[s], top[s][k].v, top[s][k].f1, top[s][k].f0, top[s][k].H, top[s][k].I);
    fclose(o);
    return 0;
}
