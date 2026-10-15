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
