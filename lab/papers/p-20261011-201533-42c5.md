# Discovery Engine: running draft

## Results

We report seven replicated findings. Each rests on two independently seeded runs, so 14 evidence runs in total. Four findings support their hypotheses or are mixed. Four pre-registered hypotheses were refuted. Unless stated otherwise, a probe "forms" or "beats control" when $R^2 - R^2_{\text{control}} \geq 0.1$. `val_mse_ratio` is model MSE divided by the MSE of constant-velocity extrapolation.

### When structure forms

**Structure related to gravity appears before the model beats the baseline.** In the 4000-step runs with 250-step checkpoints, `accel_step` already beat its control at the first trained checkpoint, step 250. Its layer-4 margins were 0.158 (seed 2) and 0.127 (seed 3). `distance` margins were 0.597 and 0.681 at layer 1. `val_mse_ratio` first fell below 0.1 only at step 1500 in both seeds (0.0858 and 0.0985), and it ended near 0.011 [f-20261011-201533-0653].

Dense checkpoints every 25 steps (seeds 4 and 5) narrow this down [f-20261011-203659-4e2f]:

- **Distance and energy** formed at step 25, while `val_mse_ratio` was still 5 to 83.
- **`accel_step`** first crossed the threshold at step 100 in both seeds (margins 0.115 and 0.108). At that point `val_mse_ratio` was still 1.96 and 1.84.
- **`val_mse_ratio` < 1** was first reported at step 175 (0.736 and 0.878).
- **Force** formed last or not at all: step 450 in seed 5, and never in seed 4.

The dense runs ended less well trained, at `val_mse_ratio` 0.141 and 0.166 [f-20261011-203659-4e2f].

![Formation trajectory of probe margins and val_mse_ratio, seed 2 (250-step checkpoints)](figure:formation/run-20261011-195457-2ec2)

![Formation trajectory of probe margins and val_mse_ratio, seed 4 (25-step checkpoints)](figure:formation/run-20261011-202258-3cf9)

In every run where timing could be resolved, force never formed before `accel_step`. Seed 2 formed both at step 250. In seed 3, force formed at 750 and `accel_step` at 250 [f-20261011-201533-0653]. In seed 5, force formed at 450 and `accel_step` at 100. In seed 4, force never formed [f-20261011-203659-4e2f].

### Force versus `accel_step`

The physical force is $|F| = m_2/r^2$. What next-step prediction needs is the per-snapshot acceleration, $|a_{\text{step}}| = (2\pi/N)^2\, a^3 / ((1+m_2)\, r^2)$. In all 14 runs, `accel_step` has the higher raw $R^2$: 0.395–0.946, against 0.154–0.750 for force (table below). Which probe has the larger *margin* over its control depends on the configuration. Control $R^2$ also varies widely between configurations, so margins in different rows should be compared with care.

| Experiment | Seed | force $R^2$ (control) | force margin | `accel_step` $R^2$ (control) | `accel_step` margin |
|---|---|---|---|---|---|
| Default $m_2$ range [f-20261011-201533-0653] | 2 | 0.738 (0.516) | 0.223 | 0.914 (0.710) | 0.204 |
| | 3 | 0.696 (0.465) | 0.230 | 0.925 (0.752) | 0.172 |
| Wide $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9] | 2 | 0.737 (0.494) | 0.243 | 0.914 (0.709) | 0.205 |
| | 3 | 0.681 (0.446) | 0.235 | 0.928 (0.753) | 0.175 |
| Dense checkpoints [f-20261011-203659-4e2f] | 4 | 0.574 (0.505) | 0.068 | 0.898 (0.742) | 0.156 |
| | 5 | 0.624 (0.521) | 0.103 | 0.904 (0.740) | 0.165 |
| OOD training runs [f-20261011-203723-b7de] | 7 | 0.738 (0.529) | 0.209 | 0.934 (0.757) | 0.178 |
| | 6 | 0.689 (0.523) | 0.166 | 0.935 (0.727) | 0.208 |
| Full-$e$ training, $e \in [0, 0.8]$ [f-20261011-211551-d718] | 9 | 0.452 (0.244) | 0.208 | 0.715 (0.378) | 0.337 |
| | 8 | 0.528 (0.216) | 0.312 | 0.707 (0.349) | 0.358 |
| Fixed $m_2 = 0.5$ [f-20261011-223000-b5f9] | 14 | 0.750 (0.560) | 0.19 | 0.924 (0.743) | 0.18 |
| | 15 | 0.732 (0.508) | 0.224 | 0.946 (0.762) | 0.18 |
| Wide $e \in [0, 0.9]$, $m_2 \in [0.01, 1]$ [f-20261011-223027-8fb3] | 16 | 0.154 (0.057) | 0.097 | 0.395 (0.175) | 0.220 |
| | 17 | 0.237 (0.131) | 0.106 | 0.503 (0.182) | 0.321 |

**Refuted: widening the mass range does not weaken the force probe** [f-20261011-203628-5cd9]. We predicted that widening $m_2$ to $[0.01, 1]$ would decouple force from `accel_step` and make the force probe fall behind. Instead, force kept the larger margin in both seeds (0.243 and 0.235, against 0.205 and 0.175). Its margins did not drop by $\geq 0.05$ relative to the default-range runs. The `accel_step` probe still worked: its margin stayed above 0.1, and it led on raw $R^2$ by 0.18–0.25. Both models beat the baseline (`val_mse_ratio` 0.0114 and 0.0119).

The margin criterion is biased against `accel_step`, whose controls are already 0.71–0.75. Measured as a fraction of available headroom, $(R^2 - R^2_{\text{control}})/(1 - R^2_{\text{control}})$, `accel_step` leads: 0.70 vs 0.48 in seed 2 and 0.71 vs 0.42 in seed 3 [f-20261011-203628-5cd9]. The refutation therefore shows only that some force-specific signal survives beyond the control.

**Refuted: missing $m_2$ information does not explain the force ceiling** [f-20261011-223000-b5f9]. With $m_2$ fixed at 0.5, force is exactly $0.5/r^2$, so a deterministic function of $r$. Even so, the force probe stayed at 0.750 (seed 14, layer 1) and 0.732 (seed 15, layer 2). The hypothesis predicted 0.85. `accel_step` reached 0.924 and 0.946 at layer 4, a raw lead of 0.17–0.21. Distance decoded at 0.87–0.89. The force ceiling of about 0.73–0.75 therefore comes from the $1/r^2$ nonlinearity, or from how the layers encode it, and not from $m_2$ being absent from the residual stream. Energy decoded unusually well in these runs: 0.940 and 0.911, against controls of 0.302 and 0.201 [f-20261011-223000-b5f9].

![Layer × checkpoint probe heatmap for force, seed 15 (m2 fixed at 0.5)](figure:heatmap/run-20261011-213802-62f9/force)

![Layer × checkpoint probe heatmap for accel_step, seed 15 (m2 fixed at 0.5)](figure:heatmap/run-20261011-213802-62f9/accel_step)

### Symbolic recovery of the law

**Refuted: the probe directions do not yield $r^{-2}$ even with wide eccentricity coverage** [f-20261011-223027-8fb3]. Seeds 16 and 17 were trained with $e \in [0, 0.9]$ and $m_2 \in [0.01, 1]$. We fit symbolic expressions to the best probe directions. In both seeds the `accel_step` fit scaled as $r^{-1}$, not $r^{-2}$:

- **Seed 16:** $0.0103\, a^2/r$, with fit error against the true law 0.71.
- **Seed 17:** $0.00277\,\big((a+0.685)^3/r - r\big)$. This is roughly $a^3$ but still $r^{-1}$, with fit error 0.62.

The force fits were $m_2/r + 0.40$ and $(m_2 + 0.60)/r$, with errors 0.90 and 0.64. No run recovered the law.

Prediction stayed strong: `val_mse_ratio` was 0.0137 and 0.0128. The `accel_step` probe still beat its control (margins 0.220 and 0.321), but its raw $R^2$ fell to 0.395 and 0.503, from 0.92–0.95 at $e \leq 0.5$. Force barely registered: margin 0.097 in seed 16, below the 0.1 threshold, and 0.106 in seed 17 [f-20261011-223027-8fb3]. These are the only runs with symbolic fits. The other twelve evidence runs have no equations [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f, f-20261011-203723-b7de, f-20261011-211551-d718, f-20261011-223000-b5f9].

![Layer × checkpoint probe heatmap for accel_step, seed 17 (e ∈ [0, 0.9])](figure:heatmap/run-20261011-215313-2e1f/accel_step)

### Where in the network each quantity is read out

- **`accel_step`** is best at layer 4 in all 14 runs.
- **`energy`** is best at layer 1 in all 14 runs.
- **`distance`** is best at layer 1 in 13 of 14 runs. The exception is seed 16, at layer 2 [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f, f-20261011-203723-b7de, f-20261011-211551-d718, f-20261011-223000-b5f9, f-20261011-223027-8fb3].
- **Force** has no stable best layer:
  - Layer 4: seed 2 (both experiments), seeds 4, 5 and 9.
  - Layer 1: seed 3 (both experiments), seeds 6, 7, 8, 14, 16 and 17.
  - Layer 2: seed 15.

`distance` margins are large throughout, for example 0.889 against a control of 0.258 in seed 2 [f-20261011-201533-0653]. Because force and `accel_step` both scale as $1/r^2$, part of the force signal may be distance information read at a different layer [f-20261011-203628-5cd9].

### Out-of-distribution transfer

**Restricted-eccentricity training (seeds 6 and 7)** [f-20261011-203723-b7de]. In-distribution `val_mse_ratio` was 0.0151 and 0.0157.

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): transfer passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× in-distribution. Force $R^2$ held at 0.75–0.77.
- **High eccentricity ($e \in [0.5, 0.8]$): transfer failed the pre-registered 10× criterion.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution. Force $R^2$ fell to 0.40 and 0.34.
- **Every two-body ratio stayed below 1.**
- **Three-body ratios of 72 and 70 are exploratory only.**

**Full-eccentricity training, $e \in [0, 0.8]$ (seeds 8 and 9)** [f-20261011-211551-d718]. The high-$e$ gap disappeared:

- Held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution (0.0178 and 0.0187). The pre-registered threshold was 3×.
- Held-out low-$e$ systems scored 0.0185 and 0.0215, so high $e$ was no worse.

The earlier failure was therefore a coverage gap. However, the force probe on held-out high-$e$ systems stays at 0.366–0.377, against 0.729–0.802 on low-$e$ systems, even though prediction error is the same on both [f-20261011-211551-d718]. This matches the earlier drop to 0.34–0.40 [f-20261011-203723-b7de]. The wide-$e$ runs show the same pattern within the training distribution: once the training range extends to $e = 0.9$, linear decodability of both force and `accel_step` collapses while prediction stays good [f-20261011-223027-8fb3].

### Null and negative results

- **Four hypotheses refuted:**
  - Widening the mass range did not make force fall behind `accel_step` [f-20261011-203628-5cd9].
  - Restricted-$e$ training did not transfer to high $e$ within 10× [f-20261011-203723-b7de]. The follow-up traced this to training coverage [f-20261011-211551-d718].
  - Fixing $m_2$ did not lift force $R^2$ toward 0.85 [f-20261011-223000-b5f9].
  - Symbolic fits gave $r^{-1}$, not $r^{-2}$ [f-20261011-223027-8fb3].
- **No explicit $1/r^2$ law has been extracted.** In the only runs with fits, every fit has the wrong exponent in $r$, with errors of 0.62–0.90 [f-20261011-223027-8fb3].
- **Force sometimes never forms.** Final margins were 0.068 in seed 4 [f-20261011-203659-4e2f] and 0.097 in seed 16 [f-20261011-223027-8fb3].
- **Probe quality and prediction quality come apart at high $e$.** At high $e$, force and `accel_step` become hard to decode linearly while `val_mse_ratio` stays near 0.013–0.019 [f-20261011-211551-d718, f-20261011-223027-8fb3].
- **"First beats baseline" is not a stable event.** Seed 5's `val_mse_ratio` rose back to 1.057 at step 225 [f-20261011-203659-4e2f].
- **No `velocity` probe results have been reported.**

### Caveats

- **Thin formation margins.** At formation, `accel_step` margins were 0.108–0.158. With a 0.12 threshold, the dense-run crossing would move to about step 175 and coincide with `val_mse_ratio` < 1 instead of preceding it [f-20261011-201533-0653, f-20261011-203659-4e2f].
- **Early probes may read input geometry.** `distance` and `energy` form while the model is still far worse than baseline. This may reflect geometry that is easy to read from the input embedding, not computation [f-20261011-203659-4e2f].
- **Gaps and missing timing in the logs.**
  - The trajectories skip several checkpoints [f-20261011-201533-0653, f-20261011-203659-4e2f].
  - Several experiments probed only the final checkpoint, so they say nothing about formation time [f-20261011-203628-5cd9, f-20261011-203723-b7de, f-20261011-211551-d718, f-20261011-223000-b5f9, f-20261011-223027-8fb3].
- **Unexplained early stops.**
  - The full-$e$ runs stopped at step 2500 of 3000 [f-20261011-211551-d718].
  - In the fixed-$m_2$ runs, seed 14 stopped at step 1500 (`val_mse_ratio` 0.063) and seed 15 at step 2500 (0.0145) [f-20261011-223000-b5f9].
- **Linear probes only.** Low $R^2$, the force ceiling and the $r^{-1}$ fits could all be artifacts of linear readout of a heavy-tailed $1/r^2$ target near pericentre [f-20261011-223000-b5f9, f-20261011-223027-8fb3].
- **No controls on the OOD evaluation sets** [f-20261011-211551-d718].
- **Limited sweep.** Two seeds per finding, a shared data seed in the wide-$e$ runs, and a single architecture ($L = 4$, $d = 64$, $N = 64$) [f-20261011-211551-d718, f-20261011-223027-8fb3].
- **Correlation in time is not use.** None of these results shows that the model *uses* the probed quantities to predict.

### Open questions

- **Nonlinear or log-target probes.** Does an MLP or $\log(\texttt{accel\_step})$ probe reach $R^2 \geq 0.8$? If so, does its fitted direction recover $r^{-2}$? The same question applies to force with $m_2$ fixed (0.85 or above) [f-20261011-223000-b5f9, f-20261011-223027-8fb3].
- **Narrow-$a$ fits.** Do symbolic fits restricted to systems with a narrow range of $a$ recover $r^{-2}$ [f-20261011-223027-8fb3]?
- **Is the force signal its own representation?** Does force-specific signal survive once `accel_step` and `distance` are regressed out [f-20261011-203628-5cd9]?
- **Is the formation ordering robust?** It should be retested with 5–10-step checkpoints, a stricter margin and more seeds [f-20261011-203659-4e2f].
- **What explains the high-$e$ probe drop?** Random controls on the OOD evaluation sets are needed, plus full-length training runs [f-20261011-211551-d718].
- **Should the margin criterion be normalised by headroom** [f-20261011-203628-5cd9]?
