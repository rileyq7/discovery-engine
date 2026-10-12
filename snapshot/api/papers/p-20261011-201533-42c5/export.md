# Discovery Engine: running draft

## Abstract

When a small transformer predicts orbits well, does its residual stream hold the physical law or a shortcut that happens to work, and when does that structure form? We train transformers on simulated REBOUND orbits ($G = M = 1$), each sampled at a fixed number of snapshots $N$ per orbit. We then fit linear probes for physical force ($m_2/r^2$), per-snapshot acceleration `accel_step` ($\propto a^3/r^2$), distance and energy, and judge each probe against randomly initialised controls. The lab has 13 replicated findings (8 supported, 5 refuted) from 30 evidence runs. The budget hard stop cut 4 of those runs short. The models beat constant-velocity extrapolation, with `val_mse_ratio` as low as 0.0032 [f-20261012-030233-ab28]. At $N = 64$, `accel_step` clears its control margin at steps 80–140, which is 40–90 steps before `val_mse_ratio` falls below 1. Force never beats its control in these runs [f-20261012-002713-8d93, f-20261012-030203-1224]. At $N = 16$ the order reverses because the baseline is weaker [f-20261012-015955-fc2b]. In a 6-layer model, linear `accel_step` $R^2$ levels off near 0.70–0.74 while error keeps falling 5.5–6.8× [f-20261012-030233-ab28]. The main negative result is that no $1/r^2$ law has been extracted: symbolic fits to the probe directions give $r^{-1}$ forms in all four cited seeds [f-20261011-223027-8fb3, f-20261011-231118-7a37].

## Methods

### Data generation

All orbits are simulated with REBOUND in units where $G = 1$ and the primary mass is $M = 1$. The gravitational force on body 2 is therefore $|F| = m_2/r^2$, where $m_2$ is the mass ratio. Varying $M$ would add nothing. Positions are sampled at fixed phase intervals, so absolute mass and time scale cannot be identified from them.

Each orbit is sampled at a fixed number of snapshots per orbit, $N$ (`steps_per_orbit`). The time between snapshots is therefore $\Delta t = P/N$, which grows with orbit size ($\Delta t^2 \propto a^3$). The model never sees physical time.

Three-body systems are hierarchical: the third body sits at 3.5–6× the inner semi-major axis. They are used only as an evaluation set for models trained on two-body systems.

The configuration schema adds these fields to the specification:

- `data.eccentricity_range`
- `data.data_seed`
- `probes.control_margin`
- `probes.patching`
- `probes.symbolic`
- `eval_sets`, for out-of-distribution evaluation

Unknown fields are rejected. The training ranges differ between the cited experiments:

- Some runs train on the default range, which reaches $e \leq 0.5$ [f-20261011-223027-8fb3].
- Some runs widen the mass ratio to $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9].
- Some runs widen eccentricity to $e \in [0, 0.8]$ [f-20261011-211551-d718].
- Some runs widen both, to $e \in [0, 0.9]$ and $m_2 \in [0.01, 1]$ [f-20261011-223027-8fb3]. Both of these runs share data seed 0.

The held-out evaluation sets are:

- low mass ratio, $m_2 \in [0.01, 0.1]$
- high eccentricity, $e \in [0.5, 0.8]$
- three-body systems [f-20261011-203723-b7de]
- held-out low-$e$ and high-$e$ splits, used for the wide-eccentricity runs [f-20261011-211551-d718]

### Tokens, model and prediction target

Each token is one time step. It holds the $xy$ positions of up to three bodies (6 numbers) in the centre-of-mass frame, with absent bodies set to zero. Because every system uses the same token format, a model trained on two bodies can be evaluated on three-body sets.

The model predicts the normalised displacement to the next step. All evidence runs use one transformer architecture: $L = 4$ layers, width $d = 64$, and $N = 64$ snapshots per orbit [f-20261011-211551-d718].

Prediction quality is reported as `val_mse_ratio`. This is the model's MSE divided by the MSE of constant-velocity extrapolation, $\hat{x}_{t+1} = 2x_t - x_{t-1}$. Both are computed in physical position units over steps $t \geq 1$ on the same held-out systems. A value below 1 means the model beats the baseline.

### Training

The lab's default first-experiment (E1) configuration trains for 5k steps at batch size 128 and learning rate $10^{-3}$. Training runs on a laptop GPU (MPS) and is kept CPU-friendly.

Checkpoint spacing varies between experiments. One formation experiment probes every 250 steps [f-20261011-201533-0653]. A dense-checkpoint experiment probes every 25 steps early in training [f-20261011-203659-4e2f].

Every finding is replicated on two training seeds.

**Budget hard stop.** A runaway guard stops a run once it reaches 2× its seed's share of `budget.max_compute_hours`. The clock starts at the first training step, so dataset generation does not count. A stopped run is checkpointed and probed at its stop step. It records `steps_completed` and `steps_planned`, and its `completion` is the fraction of planned steps it trained. Runs below 80% completion do not count toward replication.

Two of the 12 evidence runs were cut short by an earlier version of this guard. That version stopped at 1.2× the budget share and counted dataset generation against it. Both runs are in [f-20261011-211551-d718]: seed 9 stopped at completion 0.867 (step 2640 of 3000) and seed 8 at 0.9 (step 2714). Under that version the probes were taken at the last regular checkpoint, step 2500, not at the stop step. Their `val_mse_ratio` was still falling when training stopped.

### Probe targets

Linear probes read five quantities from the residual stream at every layer:

- `distance`
- relative `velocity`
- `energy`, which is constant per system
- physical `force`, $|F| = m_2/r^2$
- the per-snapshot acceleration `accel_step`:

$$|a_{\text{step}}| = \frac{F}{m_2}\,\Delta t^2 = \left(\frac{2\pi}{N}\right)^2 \frac{a^3}{(1+m_2)\,r^2}.$$

Next-step prediction needs `accel_step`, not the physical force. A weak `force` probe on its own could therefore mean the model learned a per-snapshot version of gravity, not that it learned no gravity. For this reason `force` and `accel_step` are always compared.

A unit test checks two properties of `accel_step`:

- It equals the second difference of positions, to within 1%.
- It is not proportional to the physical force across orbit sizes.

`velocity` probe results do not appear in any cited finding.

### Probe protocol and random-model controls

Probes run on 300 held-out systems, at every other time step from $t = 8$. Systems are split 70/30 into probe-train and probe-test sets, so rows from the same orbit never appear on both sides.

Each probe is a ridge regression (RidgeCV over $\alpha \in [10^{-3}, 10^{4}]$) on standardised inputs and outputs. $R^2$ is averaged across output dimensions.

Every probe has a control: the same probe fitted on a randomly initialised model of the same architecture. There is one control per layer, because a random model does not change across checkpoints. A probe result counts only if

$$R^2 - R^2_{\text{control}} \geq 0.1 \quad (\texttt{probes.control\_margin}).$$

For each target, the lab reports:

- the best layer
- its $R^2$ and control $R^2$
- `formed_at_step`, the first checkpoint at which the margin reaches the threshold (null if it never does)

The control $R^2$ for `accel_step` is high under the default ranges (0.71–0.75). This caps the margin `accel_step` can reach. One finding therefore also reports a post hoc, headroom-normalised score, $(R^2 - R^2_{\text{control}})/(1 - R^2_{\text{control}})$, but its pre-registered criterion uses the raw margin [f-20261011-203628-5cd9].

Probes on the out-of-distribution evaluation sets have no random controls. Their $R^2$ values therefore cannot be read as margins [f-20261011-203723-b7de, f-20261011-211551-d718].

Only the final checkpoint's activations are saved to disk, as float16. Activations for earlier checkpoints are recomputed from the saved weights when needed.

### Symbolic regression

When `probes.symbolic` is enabled, PySR fits a closed-form expression to the best-layer probe direction for `force` and for `accel_step`. The `accel_step` fit uses the inputs $r$, $m_2$ and $a$. Each fit reports:

- the equation
- whether the true law was recovered (`law_recovered`)
- `fit_error_vs_true_law`

Symbolic fits were run at the final checkpoint only, and only in the runs behind [f-20261011-223027-8fb3]. These are the only evidence runs with fitted equations.

### Activation patching

The pipeline also supports activation patching. Heads are ranked by a force probe on each head's contribution to the residual stream. The top three heads, plus the lowest-ranked head as a control, are each patched with that head's output on a different system, and the relative increase in next-step MSE is recorded. None of the cited findings report patching results.

## Results

The lab has 13 citable findings, all with status `replicated`. Of these, 8 support their hypothesis and 5 refute it. They rest on 30 evidence runs. The budget hard stop cut 4 of those runs short, and we flag them wherever their numbers appear. Every run uses $G = 1$ and $M = 1$, so the physical force is $|F| = m_2/r^2$. Next-step prediction instead needs the per-snapshot acceleration $|a_{\text{step}}| = (2\pi/N)^2\, a^3/((1+m_2)\, r^2)$. We therefore compare the `force` and `accel_step` probes throughout.

### Prediction quality

Trained models beat constant-velocity extrapolation by a wide margin in every configuration with full-length training:

- **Default 4-layer models:** final `val_mse_ratio` was about 0.011 [f-20261011-201533-0653], 0.0114 and 0.0119 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9], and 0.0151 and 0.0157 in the transfer experiment [f-20261011-203723-b7de].
- **Wider eccentricity:** 0.0137 and 0.0128 at $e \in [0, 0.9]$ [f-20261011-223027-8fb3], and 0.0135 and 0.0115 in the symbolic replication [f-20261011-231118-7a37].
- **6-layer, $d_{\text{model}} = 128$ model at $e \in [0, 0.85]$:** 0.0037 and 0.0032 [f-20261012-030233-ab28].
- **Short dense-checkpoint runs** ended far less converged:
  - At $N = 64$: 0.51–0.53 [f-20261012-002713-8d93], 0.43–0.47 [f-20261012-005950-42a7] and 0.45–0.48 [f-20261012-030203-1224].
  - At $N = 16$: 0.017–0.018 [f-20261012-015955-fc2b].

### When structure forms

**Supported: probe-readable structure appears early, and at $N = 64$ `accel_step` forms before the model beats the baseline.**

- **250-step checkpoints [f-20261011-201533-0653]:**
  - At the first trained checkpoint (step 250), the `accel_step` margin was already 0.158 (seed 2) and 0.127 (seed 3), and the `distance` margin was 0.597 and 0.681.
  - At that point `val_mse_ratio` was still 0.87. It first fell below 0.1 at step 1500 in both seeds.
- **25-step checkpoints [f-20261011-203659-4e2f]:**
  - `accel_step` first reached the 0.1 margin at step 100 in both seeds (0.115 and 0.108). `val_mse_ratio` was then still 1.96 and 1.84.
  - `val_mse_ratio` was first reported below 1 at step 175.
  - `distance` and `energy` formed at step 25.
- **$N = 64$, 10-step checkpoints, three seeds [f-20261012-002713-8d93]:**
  - `accel_step` formed at steps 140, 140 and 100.
  - That was 40, 40 and 90 steps before `val_mse_ratio` first dropped below 1 (steps 180, 180, 190).
  - The margin never fell back below 0.1 afterwards.
- **$N = 64$, best-of-5 controls [f-20261012-005950-42a7]:**
  - `accel_step` formed at steps 85–90.
  - That was 45–75 steps before `val_mse_ratio` first fell below 1 (steps 135–165).
- **$N = 64$, best-of-5 controls, three new seeds [f-20261012-030203-1224]:**
  - `accel_step` formed persistently at steps 90, 90 and 80 (seeds 57, 58, 56), all inside the predicted window of [70, 100].
  - `val_mse_ratio` first dropped below 1 at steps 150, 150 and 170.
  - Final layer-4 margins were 0.139, 0.144 and 0.173 (headroom-normalised 0.61, 0.54 and 0.59).

![Probe formation over training, N = 64, seed 57](figure:formation/run-20261012-022939-18f0)

**Supported: at $N = 16$ the ordering flips, but the formation step barely moves** [f-20261012-015955-fc2b].

- **Ordering:** `val_mse_ratio` dropped below 1 at step 20 in all three seeds (0.80, 0.77, 0.76). `accel_step` formed later, at steps 90, 75 and 80.
- **Formation step:** these are within 5–15 steps of the seed-matched $N = 64$ runs (85, 90, 85).
- **Interpretation:** the flip comes from the weaker constant-velocity baseline at $N = 16$. So "`val_mse_ratio` < 1" is a baseline-dependent marker and cannot time formation across $N$.

**Refuted: the `accel_step` margin clears a stricter 0.15 bar before the model beats the baseline, and the early force signal is a control artefact** [f-20261012-005950-42a7].

- **0.15 bar:** in two of three $N = 64$ seeds the margin levelled off at 0.134–0.135. In seed 48 it reached 0.150 at step 150, only 5 steps before `val_mse_ratio` < 1.
- **At $N = 16$:** the margin never reached 0.15 in any seed (maximum 0.144) [f-20261012-015955-fc2b].

**A transient force signal appears early and decays.**

- **$N = 64$, best-of-5 controls [f-20261012-005950-42a7]:**
  - The force margin was 0.166–0.189 at step 10 and decayed to about 0.00–0.02 by steps 40–50.
  - The spike comes from the trained model's force $R^2$ rising (from about 0.37–0.47 at step 0 to 0.61–0.64 at step 10) and then falling back. It is not a low draw of controls.
  - Per-checkpoint control spread was not reported.
- **$N = 64$, new seeds:** the spike reached 0.152–0.158 at step 10 and fell to about 0 by steps 40–70 [f-20261012-030203-1224].
- **$N = 16$:** a spike of 0.106–0.127 at step 10 collapsed by steps 20–30 [f-20261012-015955-fc2b].

**Caveats on timing.**

- **Thin margins at the crossing:** 0.108–0.158 [f-20261011-201533-0653, f-20261011-203659-4e2f, f-20261012-002713-8d93]. In seed 58 the crossing is at exactly 0.100, and a bar 0.02 higher would move formation by about 20 steps [f-20261012-030203-1224].
- **A loss-based definition could reverse the order.** By step 50, `val_mse_ratio` had already dropped from about 70–80 to about 6–8 [f-20261012-002713-8d93].
- **Threshold-free measures keep the $N = 64$ ordering.** Half of the final margin is reached at steps 50–65. A headroom-normalised margin of 0.5 is first reached at steps 100, 125 and 130 [f-20261012-005950-42a7].
- **Distance has not been ruled out.** Nobody has yet tested whether regressing `distance` out of `accel_step` removes the margin [f-20261012-002713-8d93].
- **Early formation is only a correlation in time.** It does not show that the model uses this structure to predict [f-20261011-201533-0653].

### Force versus `accel_step`

**Raw $R^2$.** In every evidence run, `accel_step` has the higher raw $R^2$.

**Margin over control.** Which probe leads on margin depends on the configuration:

| Experiment | Seed | force $R^2$ (control) | force margin | `accel_step` $R^2$ (control) | `accel_step` margin |
|---|---|---|---|---|---|
| Default range [f-20261011-201533-0653] | 2 | 0.738 (0.516) | 0.223 | 0.914 (0.710) | 0.204 |
| | 3 | 0.696 (0.465) | 0.230 | 0.925 (0.752) | 0.172 |
| $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9] | 2 | 0.737 (0.494) | 0.243 | 0.914 (0.709) | 0.205 |
| | 3 | 0.681 (0.446) | 0.235 | 0.928 (0.753) | 0.175 |
| 25-step checkpoints [f-20261011-203659-4e2f] | 4 | 0.574 (0.505) | 0.068 | 0.898 (0.742) | 0.156 |
| | 5 | 0.624 (0.521) | 0.103 | 0.904 (0.740) | 0.165 |
| Transfer, in-distribution [f-20261011-203723-b7de] | 7 | 0.738 (0.529) | 0.209 | 0.934 (0.757) | 0.178 |
| | 6 | 0.689 (0.523) | 0.166 | 0.935 (0.727) | 0.208 |
| $e \in [0, 0.8]$, stopped early [f-20261011-211551-d718] | 9 | 0.452 (0.244) | 0.208 | 0.715 (0.378) | 0.337 |
| | 8 | 0.528 (0.216) | 0.312 | 0.707 (0.349) | 0.358 |
| $m_2 = 0.5$ fixed, stopped early [f-20261011-223000-b5f9] | 14 | 0.750 (0.560) | 0.19 | 0.924 (0.743) | 0.18 |
| | 15 | 0.732 (0.508) | 0.224 | 0.946 (0.762) | 0.18 |
| $e \in [0, 0.9]$ [f-20261011-223027-8fb3] | 16 | 0.154 (0.057) | 0.097 | 0.395 (0.175) | 0.220 |
| | 17 | 0.237 (0.131) | 0.106 | 0.503 (0.182) | 0.321 |
| Symbolic replication [f-20261011-231118-7a37] | 24 | 0.692 (0.451) | 0.241 | 0.831 (0.487) | 0.344 |
| | 26 | 0.660 (0.395) | 0.265 | 0.881 (0.581) | 0.299 |
| $N = 64$, 10-step [f-20261012-002713-8d93] | 45 | 0.544 (0.504) | 0.040 | 0.884 (0.775) | 0.109 |
| | 44 | 0.190 (0.190) | 0.000 | 0.905 (0.786) | 0.119 |
| | 46 | 0.543 (0.482) | 0.062 | 0.898 (0.740) | 0.158 |
| $N = 64$, best-of-5 [f-20261012-005950-42a7] | 47 | 0.485 (0.447) | 0.039 | 0.904 (0.770) | 0.134 |
| | 49 | 0.486 (0.440) | 0.046 | 0.884 (0.750) | 0.134 |
| | 48 | 0.523 (0.464) | 0.059 | 0.892 (0.725) | 0.167 |
| $N = 64$, best-of-5, new seeds [f-20261012-030203-1224] | 57 | 0.529 (0.450) | 0.079 | 0.909 (0.769) | 0.139 |
| | 58 | 0.564 (0.494) | 0.070 | 0.875 (0.731) | 0.144 |
| | 56 | 0.585 (0.535) | 0.051 | 0.882 (0.709) | 0.173 |
| $N = 16$, best-of-5 [f-20261012-015955-fc2b] | 47 | 0.497 (0.467) | 0.030 | 0.909 (0.793) | 0.117 |
| | 48 | 0.543 (0.488) | 0.055 | 0.897 (0.752) | 0.144 |
| | 49 | 0.578 (0.456) | 0.123 | 0.907 (0.770) | 0.137 |
| 6-layer, $d = 128$, $e \in [0, 0.85]$ [f-20261012-030233-ab28] | 56 | 0.566 (0.265) | 0.300 | 0.735 (0.229) | 0.506 |
| | 57 | 0.516 (0.183) | 0.333 | 0.705 (0.239) | 0.465 |

**Supported: in short dense-checkpoint runs, the model has a per-snapshot acceleration code but mostly no lasting physical-force code** [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-030203-1224, f-20261012-015955-fc2b].

- **`accel_step`** beats its control at the final checkpoint in every one of these runs.
- **Force** fails its required margin in every $N = 64$ dense run.
- **The exception:** at $N = 16$, force beats its control only in seed 49, where it formed at step 125.
- **Longer runs:** force keeps a margin of about 0.17–0.33 [f-20261011-201533-0653, f-20261011-203723-b7de, f-20261011-231118-7a37, f-20261012-030233-ab28]. However, force and `accel_step` are collinear through $1/r^2$, so part of that signal may be distance information [f-20261011-203628-5cd9].

![Layer × checkpoint probe heatmap for force, N = 64 seed 44 (force never beats control)](figure:heatmap/run-20261011-234250-6e25/force)

![Layer × checkpoint probe heatmap for accel_step, N = 64 seed 44](figure:heatmap/run-20261011-234250-6e25/accel_step)

**Refuted: widening the mass range makes force fall behind `accel_step`** [f-20261011-203628-5cd9].

- **Result:** with $m_2 \in [0.01, 1]$, force kept the larger margin in both seeds.
- **The margin criterion favours force.** `accel_step`'s control $R^2$ of 0.71–0.75 caps its possible margin. Measured as a fraction of headroom, `accel_step` leads: 0.70 vs 0.48 in seed 2 and 0.71 vs 0.42 in seed 3.

**Refuted: fixing $m_2$ lifts the force probe** [f-20261011-223000-b5f9].

- **Prediction:** with $m_2 = 0.5$, force $R^2$ would reach 0.85.
- **Result:** force $R^2$ reached only 0.750 and 0.732, while `distance` was decoded at 0.87–0.89. The force ceiling therefore comes from the $1/r^2$ nonlinearity or how it is encoded, not from missing $m_2$ information.
- **Both runs were cut short by the budget hard stop.**
  - Seed 14 stopped at step 1500 (completion 0.55, `val_mse_ratio` 0.063). It is below the lab's 80% bar and does not count as an agreeing seed.
  - Seed 15 stopped at step 2500 (completion 0.917). It was the better-trained model and had the slightly lower force $R^2$.

**Where each quantity is read out.**

- **`accel_step`** is best at the last layer in every evidence run: layer 4 in the 4-layer models and layer 6 in the 6-layer models [f-20261012-002713-8d93, f-20261012-030233-ab28].
- **`distance` and `energy`** are best at layer 1 or 2 [f-20261011-223027-8fb3, f-20261012-002713-8d93, f-20261012-015955-fc2b].
- **Force** has no stable best layer. It peaks at layers 0 through 4 depending on the seed, and drifts across layers 1–6 in the 6-layer model [f-20261011-201533-0653, f-20261011-223000-b5f9, f-20261012-002713-8d93, f-20261012-030233-ab28].

### Linear decodability stops tracking prediction quality

**Supported: in a larger model, the linear `accel_step` code levels off while prediction keeps improving** [f-20261012-030233-ab28].

- **Plateau:** in the 6-layer, $d_{\text{model}} = 128$ model at $e \in [0, 0.85]$, layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 (seed 56) and from 0.683 to 0.705 (seed 57) after step 2000. It never reached 0.8.
- **Prediction kept improving:** over the same steps `val_mse_ratio` fell 6.8× (0.0253 to 0.0037) and 5.5× (0.0177 to 0.0032).
- **Margins:** at step 4000, `accel_step` beats its control by 0.506 and 0.465 (control spread ≤ 0.015). Force beats its control by 0.300 and 0.333, with a headroom-normalised margin of 0.41 in both seeds, against 0.66 and 0.61 for `accel_step`.
- **Limits:** $R^2$ still creeps up by about 0.011–0.012 per 1000 steps, and checkpoints are 500 steps apart. This result does not show whether a nonlinear `accel_step` code exists.

![Layer × checkpoint probe heatmap for accel_step, 6-layer d = 128 model, seed 56](figure:heatmap/run-20261012-023515-4712/accel_step)

### Out-of-distribution transfer

**Refuted: training on restricted eccentricity transfers to high eccentricity within 10×** [f-20261011-203723-b7de].

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× the in-distribution value. The best force $R^2$ held at 0.75–0.77.
- **High eccentricity ($e \in [0.5, 0.8]$): failed.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution, and force $R^2$ fell to 0.40 and 0.34.
- **Every two-body ratio stayed below 1.** Three-body ratios of 72 and 70 are exploratory only.
- **OOD probes have no random controls.**

**Supported: training on $e \in [0, 0.8]$ removes the prediction gap** [f-20261011-211551-d718].

- **Gap closed:** held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution, well under the 3× threshold.
- **Early stop:** the budget hard stop cut both runs short at step 2500, with completion 0.867 (seed 9) and 0.9 (seed 8). Both are above the 80% bar, and `val_mse_ratio` was still falling when they stopped.
- **Probe gap remained:** on high-$e$ systems the force probe reached only 0.366–0.377, against 0.729–0.802 on low-$e$ systems, even though prediction error was the same on both.

### Symbolic recovery of the law

**Refuted: wide eccentricity coverage yields $r^{-2}$ from the probe directions** [f-20261011-223027-8fb3].

- **`accel_step` fits ($e \in [0, 0.9]$):**
  - Seed 16: $0.0103\,a^2/r$, with fit error 0.71.
  - Seed 17: $0.00277\,\big((a+0.685)^3/r - r\big)$, with fit error 0.62.
- **Force fits:** $m_2/r + 0.40$ and $(m_2 + 0.60)/r$.
- **Weak probes:** `accel_step` $R^2$ was only 0.395 and 0.503, so these fits describe a weak linear approximation.

**Supported: the linearly decodable gravity direction has an $r^{-1}$ orbit-scale form** [f-20261011-231118-7a37].

- **Better probes, same result:** with `accel_step` $R^2$ of 0.831 and 0.881, the fits were still $r^{-1}$.
  - Seed 24: $a^2/r - 0.55a + 0.076r$ (scaled by 0.0134), with fit error 0.245.
  - Seed 26: $0.0136\,(a^2/r - 0.56a) + 0.0015$, with fit error 0.201.
- **Force fits** were $0.81/r$ and $0.80/r$.
- **Caveats:**
  - Seed 26 only just clears the fit-error criterion.
  - The shape $a^2/r - c\,a$ with $c \approx 0.55$ resembles the vis-viva combination $(v\,dt)^2/a$, but this is not established.
  - The symbolic-regression pipeline has not been checked on the true `accel_step` target. If it also returns $r^{-1}$ there, the result describes the fitter, not the model.

![Layer × checkpoint probe heatmap for accel_step, seed 24](figure:heatmap/run-20261011-225359-5030/accel_step)

### Summary of null and negative results

- **Five hypotheses were refuted:**
  - Wider $m_2$ did not make force fall behind `accel_step` [f-20261011-203628-5cd9].
  - Restricted-$e$ training did not transfer to high $e$ within 10× [f-20261011-203723-b7de]. Later runs traced this to training coverage [f-20261011-211551-d718].
  - Fixing $m_2$ did not lift force $R^2$ [f-20261011-223000-b5f9].
  - Wide-$e$ training did not yield an $r^{-2}$ fit [f-20261011-223027-8fb3].
  - At $N = 64$, the `accel_step` margin did not clear a 0.15 bar before the model beat the baseline, and the early force spike survived best-of-5 controls [f-20261012-005950-42a7].
- **No $1/r^2$ law has been extracted.** Every symbolic fit so far gives $r^{-1}$ [f-20261011-223027-8fb3, f-20261011-231118-7a37].
- **Force usually ends without a margin in short runs:**
  - Seed 4 [f-20261011-203659-4e2f] and seed 16 [f-20261011-223027-8fb3].
  - Every $N = 64$ dense run [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-030203-1224].
  - Two of three $N = 16$ runs [f-20261012-015955-fc2b].
- **Linear decodability and prediction come apart:**
  - At high eccentricity [f-20261011-211551-d718, f-20261011-223027-8fb3].
  - At $N = 16$, where the model beats the baseline well before `accel_step` forms [f-20261012-015955-fc2b].
  - In the 6-layer model, where `accel_step` $R^2$ levels off while error keeps falling [f-20261012-030233-ab28].
- **Not yet reported:** `velocity` probe results and activation-patching results.
- **Limits throughout:**
  - All probes are linear.
  - Formation steps depend on the 0.1 threshold, which sits close to the `accel_step` ceiling in the 4-layer models because their control $R^2$ is 0.71–0.79.
  - The budget hard stop cut four evidence runs short. One of them (seed 14, completion 0.55) is below the replication bar [f-20261011-211551-d718, f-20261011-223000-b5f9].
