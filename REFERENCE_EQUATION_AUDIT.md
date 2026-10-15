# Reference-liquid equation audit

AI-written, 2026-10-15. Source: Johannsmann et al. (2025), DOI
10.1002/adts.202401373.

| Model element | Source | Result |
|---|---|---|
| Moments and linear equilibrium | Eqs. 11, 18 | Matches |
| TRT rates and collision | Eqs. 19, 75 | Fixed stored-rate inversion and asymmetric coefficient |
| Empirical second-order term | Section 6.2 | Fixed placement: $+(5/2)\omega^2h^{eq}$ |
| Moving QCM bounce-back and force | Eqs. 12--15 | Matches |
| Uniform top impedance | Eqs. A34--A36 | Matches |
| Stationary direct solve | Section 6.3 | Matches the same discrete update |

The paper states that the second-order correction is empirical and lacks a
consistent Chapman--Enskog derivation. Loaded-domain collision uses the old
placement and must be audited separately before this branch is merged.

The stability diagnostic compares periodic and QCM/impedance-boundary updates,
each with and without the empirical second-order correction. This separates
bulk amplification from boundary-induced amplification.

## Coefficient sensitivity

AI-written preliminary audit, 2026-10-15. The reduced QCM/impedance reference
was directly solved for $c\omega^2h^{eq}$ with $c=-5/2,+5/2,-1/2,0$ at
$\Delta x=2.5,1.25,0.625$ nm. Relative complex-response errors against
$\Delta f/n=i f_0\sqrt{i\omega\rho\eta}/(\pi Z_q)$, using water
($\rho=1000$ kg m$^{-3}$, $\eta=1$ mPa s), were:

| $\Delta x$ (nm) | $c=-5/2$ | $c=+5/2$ | $c=-1/2$ | $c=0$ |
|---:|---:|---:|---:|---:|
| 2.5 | 14.31% | 0.376% | 8.62% | 7.19% |
| 1.25 | 3.61% | 0.0235% | 2.17% | 1.81% |
| 0.625 | 0.904% | 0.00147% | 0.543% | 0.452% |

The $+5/2$ coefficient gives the closest continuum response, while its
iteration remains weakly unstable ($\rho_{op}=1.000033$ at $0.625$ nm).
The $-5/2$ case is stable there ($\rho_{op}=0.999968$) but less accurate.
This is a reduced planar-reference result, not a loaded-cylinder validation.
The local Python environment lacked a Numba-compatible NumPy; the reduced
collision function was executed as ordinary Python, with direct-solve residuals
below $1.2\times10^{-16}$.
