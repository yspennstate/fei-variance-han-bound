# A variance form of Han's bound on the Fourier entropy

Yitzchak Shmalo

This repository contains a short paper on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai,
together with the scripts behind every numerical value in it.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

For every Boolean function f on n variables,

    H(f) <= c0 I(f) + sum_k Inf_k(f) log2(1 + Var(f)/Inf_k(f)),    c0 = 2 + 3/ln 2 = 6.3281...

This is Han's bound with 1/Inf_k replaced by Var(f)/Inf_k inside the logarithm. It implies Han's bound with one
more unit in the constant, gives H(f) <= (c0 + log2(1 + 1/eps)) I(f) when every relevant influence is at least
eps Var(f), and gives H(f) <= (c0 + 1 + H(pi)) I(f) with pi_k = Inf_k/I. For AND_k the new bound is below
7.92 I(f) for every k, while Han's bound is (k + 1 + 3/ln 2) I(f).

The proof bounds the entropy drop of one coordinate by Inf_i (c0 + log2(1 + Var/Inf_i)), using the decomposition of
the drop into an erasure entropy and a Jensen-Shannon divergence
([one-coordinate note](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)), and sums the drops
along a decision tree. The sum along any adaptive tree gives a stronger statement (the adaptive theorem in Section 3 of the paper).

## Code

The scripts need Python 3 and NumPy; the census needs a C compiler. Saved outputs are next to the scripts.

* `code/check_var_han.py` (output `check_var_han_out.txt`): the main bound on 1720 random functions on 2 to 10
  variables, and the AND_k table against Han's bound for k = 2..14.
* `code/check_step_tree.py` (output `check_step_tree_out.txt`): the drop D_i, the erasure entropy E_i and the
  Jensen-Shannon term J_i computed separately on 4767 coordinate steps, the identity D_i = E_i + J_i, the two
  one-coordinate lemmas, and the adaptive theorem along 928 random adaptive trees.
* `census/mono_census.c`: exhaustive census of monotone Boolean functions on n = 6 and n = 7 variables, one
  representative per symmetry orbit of the first n - 1 variables, with the weighted count checked against the
  Dedekind numbers 7,828,354 and 2,414,682,040,998. HBvar and HBhan are the ratios of H(f) to the right sides of the
  main theorem and of Han's bound. Usage `mono_census K P p outfile` (n = K + 1, part p of P).
* `census/merge_census.py`: merges the part files, checks the count, and recomputes each leading function from its
  truth table with a separate floating-point transform.
* `census/census_n7_merged.txt`: the merged n = 7 top lists (all statistics, with recomputed values).
  `census/census_n6_ratio_top.txt`: the ten largest HBvar and HBhan values for n = 6
  (columns: statistic, value, f0, f1 as truth tables of the two restrictions in the last variable, H, I).

## Related papers

* [The one-coordinate step in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)
* [Good coordinates in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-good-coordinates)
* [Four equivalent forms of the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-equivalent-forms)
* [A filtration bound for the erasure entropies of Boolean functions](https://github.com/yspennstate/fei-filtration-witness)
* [Erasure entropies of sparse linear DNFs](https://github.com/yspennstate/fei-sparse-linear-dnf)
* [The Fourier entropy of linear monotone DNFs on few variables](https://github.com/yspennstate/fei-linear-dnf-census)

## License

MIT, see [LICENSE](LICENSE).
