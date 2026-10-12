# Discovery Engine: running draft

## Abstract

When a small transformer predicts orbits well, does its residual stream hold the physical law or a task-specific shortcut, and when does that structure form? We train transformers on REBOUND orbits sampled at a fixed number of snapshots per orbit. We then probe them linearly, judging each probe against randomly initialised controls. Probe targets are physical force $|F| = m_2/r^2$ and the per-snapshot acceleration $(F/m_2)\,dt^2$ that next-step prediction actually needs. The lab has 16 replicated findings: 10 supported and 6 refuted, from 39 runs, 4 of them cut short by the budget stop. Per-snapshot acceleration is decodable at the last layer ($R^2$ 0.884–0.905 at $N=64$) [f-20261012-002713-8d93]. It forms at steps 60–140 [f-20261012-002713-8d93, f-20261012-044142-d490, f-20261012-034018-5526], often before the model beats constant-velocity extrapolation. That ordering flips at $N=16$, where the baseline is weaker [f-20261012-015955-fc2b]. Physical force usually fails its control in short runs, apart from a brief spike at step 10 [f-20261012-005950-42a7]. The main negative result is that no $1/r^2$ law has been recovered: symbolic fits give $r^{-1}$ forms such as $a^2/r - 0.55a$ [f-20261011-223027-8fb3, f-20261011-231118-7a37]. In a larger model, linear $R^2$ levels off near 0.70–0.74 while prediction keeps improving [f-20261012-030233-ab28].

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

The lab has 16 citable findings, and all 16 are replicated. Ten support their hypothesis and six refute it. Together they rest on 39 evidence runs. The budget hard stop cut 4 of those runs short, and they are flagged where their numbers are used. Unless stated otherwise, every number below is a linear-probe $R^2$ on held-out systems, compared against the best of several randomly initialised controls.

### When structure forms

**Supported: probe-readable gravity structure appears early in training** [f-20261011-201533-0653, f-20261011-203659-4e2f, f-20261012-002713-8d93].

- **Coarse checkpoints (every 250 steps).** In both seeds, the probes already beat their controls at the first trained checkpoint, step 250 [f-20261011-201533-0653].
  - The `accel_step` margin at layer 4 was 0.158 (seed 2) and 0.127 (seed 3).
  - `val_mse_ratio` did not fall below 0.1 until step 1500 (0.0858 and 0.0985).
  - The `accel_step` margins are only just over the 0.1 bar, so they are fragile.
- **Checkpoints every 25 steps.** The `accel_step` margin first reached 0.1 at step 100 in both seeds (0.115 and 0.108). At that point `val_mse_ratio` was still 1.96 and 1.84, and it was first reported below 1 at step 175 [f-20261011-203659-4e2f].
- **Checkpoints every 10 steps, $N = 64$.** `accel_step` crossed 0.1 at steps 140, 140 and 100, which is 40, 40 and 90 steps before `val_mse_ratio` < 1 (steps 180, 180, 190). It never dropped back below 0.1 afterwards [f-20261012-002713-8d93].
- **New seeds 62–64 at $N = 64$, $e \in [0, 0.5]$, best-of-5 controls.** The margin first held at or above 0.1 for good at steps 90, 100 and 100. It first reached half its final margin at steps 80, 60 and 90 [f-20261012-044142-d490].
  - Formation came while `val_mse_ratio` was still 1.7–2.1, which is about 2.0–2.9% of each run's step-0 value.
  - That is 40–60 steps before the model beat the baseline (steps 140–160).
  - Two of the six timing values sit exactly on the edge of the pre-registered window.

**Refuted: at $N = 64$ the `accel_step` margin clears a stricter 0.15 bar before the model beats the baseline, and the early force spike is a control artefact** [f-20261012-005950-42a7].

- **The ordering holds at the default bar.** `accel_step` reached 0.1 at steps 85–90, which is 45–75 steps before `val_mse_ratio` < 1 (steps 135–165).
- **The 0.15 bar fails.** In 2 of 3 seeds the margin levelled off at 0.134–0.135. In seed 48 it reached 0.150 at step 150, only 5 steps before `val_mse_ratio` < 1.
- **The early force spike is real.** It survives best-of-5 controls in all three seeds. The force margin was 0.166–0.189 at step 10 and decayed to about 0.00–0.02 by steps 40–50.

**Formation tracks training steps, not loss reduction or beating the baseline.**

- **$N = 16$** [f-20261012-015955-fc2b]. `val_mse_ratio` fell below 1 at step 20 in all three seeds, but `accel_step` formed only at steps 90, 75 and 80. Those steps are within 5–15 steps of the seed-matched $N = 64$ runs, so the "forms before beating the baseline" ordering flips while the formation step barely moves. "`val_mse_ratio` < 1" is therefore a baseline-dependent marker.
- **$N = 64$, $e \in [0, 0.85]$** [f-20261012-034018-5526].
  - `accel_step` formed at steps 60, 60 and 70, and reached half its final margin at steps 70, 80 and 90.
  - This happened at about 8–10% of the step-0 loss, against 2.0–2.9% at $e \le 0.5$ [f-20261012-044142-d490].
- **$N = 256$, $e \in [0, 0.85]$** [f-20261012-034134-5e82].
  - `accel_step` formed at steps 60, 70 and 60, and reached half its final margin at step 70 in every seed.
  - This happened even though the model never beat constant-velocity extrapolation (final `val_mse_ratio` 2.03–2.37).
- **Refuted: halving the learning rate roughly doubles the formation step** [f-20261012-044258-ab88].
  - At lr = 0.0005, the 0.1 bar was crossed and held at steps 70, 80 and 100 (seeds 59–61), against 60/60/70 at lr = 0.001. That is a delay of +10/+20/+30 steps, not the roughly 2× (≥ 100 in every seed) that proportional scaling predicts.
  - Training did slow down: final `val_mse_ratio` was about 0.22 here against about 0.13 at lr = 0.001.
  - Formation therefore shifts with lr, but less than proportionally.

![Formation chart for N = 64, e ∈ [0, 0.5], seed 64](figure:formation/run-20261012-043022-144a)

**Caveats on timing.**

- **Distance and energy form first.** In every dense-checkpoint run, `distance` and `energy` form before `accel_step`, often by steps 10–40 [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-044142-d490].
- **`accel_step` may be a nonlinear read of distance.** Nobody has yet tested whether regressing `distance` out of `accel_step` removes its margin [f-20261012-002713-8d93].
- **Thin margins at $e \le 0.5$.** The `accel_step` controls there are 0.72–0.79, so margins sit close to the 0.1 bar and formation steps depend on the threshold [f-20261012-002713-8d93, f-20261012-015955-fc2b].
- **This is correlation in time, not use.** Early formation does not show that the model uses this structure to predict [f-20261011-201533-0653].

### Force versus `accel_step`

**Raw $R^2$.** `accel_step` has the higher raw $R^2$ in every evidence run.

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
| $N = 64$, $e \le 0.5$, seeds 62–64 [f-20261012-044142-d490] | 64 | 0.539 (0.506) | 0.033 | 0.891 (0.724) | 0.167 |
| | 62 | 0.611 (0.531) | 0.080 | 0.891 (0.746) | 0.144 |
| | 63 | 0.575 (0.514) | 0.061 | 0.902 (0.742) | 0.160 |
| $N = 16$, best-of-5 [f-20261012-015955-fc2b] | 47 | 0.497 (0.467) | 0.030 | 0.909 (0.793) | 0.117 |
| | 48 | 0.543 (0.488) | 0.055 | 0.897 (0.752) | 0.144 |
| | 49 | 0.578 (0.456) | 0.123 | 0.907 (0.770) | 0.137 |
| 6-layer, $d = 128$, $e \in [0, 0.85]$ [f-20261012-030233-ab28] | 56 | 0.566 (0.265) | 0.300 | 0.735 (0.229) | 0.506 |
| | 57 | 0.516 (0.183) | 0.333 | 0.705 (0.239) | 0.465 |
| $N = 64$, $e \in [0, 0.85]$ [f-20261012-034018-5526] | 59 | 0.318 (0.206) | 0.112 | 0.570 (0.282) | 0.287 |
| | 60 | 0.250 (0.156) | 0.094 | 0.594 (0.333) | 0.261 |
| | 61 | 0.279 (0.183) | 0.095 | 0.507 (0.248) | 0.258 |
| $N = 64$, $e \in [0, 0.85]$, lr 0.0005 [f-20261012-044258-ab88] | 59 | 0.284 (0.206) | 0.078 | 0.535 (0.282) | 0.253 |
| | 60 | 0.219 (0.154) | 0.065 | 0.590 (0.333) | 0.257 |
| | 61 | 0.255 (0.183) | 0.072 | 0.492 (0.248) | 0.244 |
| $N = 256$, $e \in [0, 0.85]$ [f-20261012-034134-5e82] | 59 | 0.097 (0.093) | 0.004 | 0.619 (0.351) | 0.268 |
| | 60 | 0.144 (0.113) | 0.031 | 0.543 (0.333) | 0.210 |
| | 61 | 0.108 (0.084) | 0.024 | 0.464 (0.238) | 0.226 |

**Supported: in short dense-checkpoint runs, the model has a per-snapshot acceleration code but mostly no lasting physical-force code** [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-034018-5526, f-20261012-034134-5e82].

- **`accel_step`** beats its control at the final checkpoint in every one of these runs.
- **Force** fails its required margin in every $N = 64$ run at $e \le 0.5$, in every $N = 256$ run, and in every lr = 0.0005 run [f-20261012-044258-ab88].
- **Exceptions:** force clears the bar only in $N = 16$ seed 49, where it formed at step 125 [f-20261012-015955-fc2b], and in high-$e$ $N = 64$ seed 59, where it formed at step 400 [f-20261012-034018-5526].
- **Transient early force signal.** A force margin of 0.148–0.159 at step 10 that falls to about 0 by step 40 recurs at $N = 64$ [f-20261012-044142-d490]. The same pattern appears at $N = 16$ (0.106–0.127) [f-20261012-015955-fc2b].
- **Longer runs keep a force margin** of about 0.17–0.33 [f-20261011-201533-0653, f-20261011-203723-b7de, f-20261011-231118-7a37, f-20261012-030233-ab28]. However, force and `accel_step` are collinear through $1/r^2$, so part of that signal may be distance information [f-20261011-203628-5cd9].

![Layer × checkpoint probe heatmap for force, N = 64 seed 44 (force never beats control)](figure:heatmap/run-20261011-234250-6e25/force)

![Layer × checkpoint probe heatmap for accel_step, N = 256, e ∈ [0, 0.85], seed 59](figure:heatmap/run-20261012-033409-533d/accel_step)

**At high eccentricity, the larger margins come from lower controls, not better decoding.**

- At $e \in [0, 0.85]$, the `accel_step` control drops to 0.24–0.35, while raw `accel_step` $R^2$ is only 0.46–0.62. At $e \le 0.5$ the raw $R^2$ is 0.85–0.95 [f-20261012-034018-5526, f-20261012-034134-5e82].
- The low-$e$ runs used for this comparison were not matched on mass range or seeds [f-20261012-034018-5526].

**Refuted: widening the mass range makes force fall behind `accel_step`** [f-20261011-203628-5cd9].

- **Result:** with $m_2 \in [0.01, 1]$, force kept the larger raw margin in both seeds.
- **Measured as a fraction of headroom, $(R^2 - \text{control})/(1 - \text{control})$, `accel_step` leads:** 0.70 vs 0.48 in seed 2 and 0.71 vs 0.42 in seed 3. The raw-margin criterion favours force, because the `accel_step` control of 0.71–0.75 caps its margin.

**Refuted: fixing $m_2$ lifts the force probe** [f-20261011-223000-b5f9].

- **Prediction and result:** with $m_2 = 0.5$, so that $|F| = 0.5/r^2$ exactly, force $R^2$ was predicted to reach 0.85. It reached only 0.750 and 0.732, while `distance` was decoded at 0.87–0.89. The force ceiling therefore comes from the $1/r^2$ nonlinearity or how it is encoded, not from missing $m_2$ information.
- **Both runs were cut short by the budget hard stop.**
  - Seed 14 stopped at step 1500 (completion 0.55). It is below the 80% replication bar.
  - Seed 15 stopped at step 2500 (completion 0.917). It was the better-trained model and had the slightly lower force $R^2$.

**Where each quantity is read out.**

- **`accel_step`** is best at the last layer in every evidence run: layer 4 in the 4-layer models and layer 6 in the 6-layer model [f-20261012-002713-8d93, f-20261012-030233-ab28, f-20261012-034134-5e82].
- **`distance`** is best at layer 1 or 2 [f-20261011-223027-8fb3, f-20261012-002713-8d93, f-20261012-015955-fc2b].
- **Force** has no stable best layer. It ranges from layer 0 to layer 4 in the 4-layer models and drifts across layers 1–6 in the 6-layer model [f-20261011-201533-0653, f-20261011-223000-b5f9, f-20261012-002713-8d93, f-20261012-030233-ab28].

### Linear decodability stops tracking prediction quality

**Supported: in a larger model, the linear `accel_step` code levels off while prediction keeps improving** [f-20261012-030233-ab28].

- **Plateau:** in the 6-layer, $d_{\text{model}} = 128$ model at $e \in [0, 0.85]$, layer-6 `accel_step` $R^2$ rose after step 2000 only from 0.710 to 0.735 (seed 56) and from 0.683 to 0.705 (seed 57). It never reached 0.8.
- **Prediction kept improving:** over the same steps `val_mse_ratio` fell 6.8× and 5.5×, to 0.0037 and 0.0032.
- **Limits:** $R^2$ still creeps up by about 0.011–0.012 per 1000 steps, and checkpoints are 500 steps apart. No nonlinear probe has been tried.

![Layer × checkpoint probe heatmap for accel_step, 6-layer d = 128 model, seed 56](figure:heatmap/run-20261012-023515-4712/accel_step)

### Out-of-distribution transfer

**Refuted: a model trained on restricted eccentricity transfers to high eccentricity within 10× of its in-distribution error** [f-20261011-203723-b7de].

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× in-distribution.
- **High eccentricity ($e \in [0.5, 0.8]$): failed.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution, and force $R^2$ fell to 0.40 and 0.34.
- **Every two-body ratio stayed below 1.** Three-body ratios of 72 and 70 are exploratory only.
- **No controls out of distribution.** The OOD probes have no random controls.

**Supported: training on $e \in [0, 0.8]$ removes the prediction gap** [f-20261011-211551-d718].

- **Gap closed:** held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution.
- **Budget stop:** both runs were cut short at step 2500, with completion 0.867 (seed 9) and 0.9 (seed 8). Both are above the 80% bar, but `val_mse_ratio` was still falling when they stopped.
- **Probe gap remained:** the force probe reached only 0.366–0.377 on held-out high-$e$ systems, against 0.729–0.802 on held-out low-$e$ systems, even though prediction error was the same on both.

### Symbolic recovery of the law

**Refuted: wide eccentricity coverage yields an $r^{-2}$ law from the probe directions** [f-20261011-223027-8fb3].

- **`accel_step` fits at $e \in [0, 0.9]$:** $0.0103\,a^2/r$ (fit error 0.71) and $0.00277\,\big((a+0.685)^3/r - r\big)$ (fit error 0.62).
- **Force fits:** $m_2/r + 0.40$ and $(m_2 + 0.60)/r$.
- **Weak probes:** `accel_step` $R^2$ was only 0.395 and 0.503, so the fits describe a weak linear approximation.

**Supported: the linearly decodable gravity direction has an $r^{-1}$ orbit-scale form** [f-20261011-231118-7a37].

- **Better probes, same exponent:** with `accel_step` $R^2$ of 0.831 and 0.881, the fits were $a^2/r - 0.55a + 0.076r$ (scaled by 0.0134, fit error 0.245) and $0.0136\,(a^2/r - 0.56a) + 0.0015$ (fit error 0.201). The true law is $\propto a^3/r^2$.
- **Force fits** were $0.81/r$ and $0.80/r$.
- **Not yet checked:** the symbolic-regression pipeline has not been run on the true `accel_step` target. If it also returns $r^{-1}$ there, the result describes the fitter, not the model.

![Layer × checkpoint probe heatmap for accel_step, seed 24](figure:heatmap/run-20261011-225359-5030/accel_step)

### Summary of null and negative results

- **Six hypotheses were refuted:**
  - Wider $m_2$ did not make force fall behind `accel_step` on raw margin [f-20261011-203628-5cd9].
  - Restricted-$e$ training did not transfer to high $e$ within 10× [f-20261011-203723-b7de]. Later runs traced this to training coverage [f-20261011-211551-d718].
  - Fixing $m_2$ did not lift force $R^2$ [f-20261011-223000-b5f9].
  - Wide-$e$ training did not yield an $r^{-2}$ fit [f-20261011-223027-8fb3].
  - At $N = 64$, `accel_step` did not clear a 0.15 bar before the model beat the baseline, and the early force spike survived best-of-5 controls [f-20261012-005950-42a7].
  - Halving the learning rate did not proportionally delay formation [f-20261012-044258-ab88].
- **No $1/r^2$ law has been extracted.** Every symbolic fit so far gives $r^{-1}$ [f-20261011-223027-8fb3, f-20261011-231118-7a37].
- **A physical-force code is usually absent in short runs.** In those runs only `accel_step` reliably beats its control [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-034018-5526, f-20261012-034134-5e82, f-20261012-044258-ab88].
- **Probe structure and prediction quality come apart.** This shows at high eccentricity [f-20261011-211551-d718, f-20261011-223027-8fb3], at $N = 16$ [f-20261012-015955-fc2b], at $N = 256$ [f-20261012-034134-5e82], and in the 6-layer model [f-20261012-030233-ab28].
- **Not yet reported:** `velocity` probe results, activation-patching results, and any test of whether the `accel_step` code is causally used.
- **Limits throughout:**
  - All probes are linear.
  - At $e \le 0.5$, formation steps depend on a 0.1 threshold that sits close to the `accel_step` ceiling.
  - Most results come from a single architecture and data seed.
  - The budget hard stop cut four evidence runs short. Seed 14 (completion 0.55) is below the replication bar [f-20261011-211551-d718, f-20261011-223000-b5f9].
