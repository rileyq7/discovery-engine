# Discovery Engine: running draft

## Abstract

When a small transformer predicts orbits well, does its residual stream hold the physical law $|F| = m_2/r^2$, or heuristics that happen to work? We train transformers on REBOUND two-body orbits sampled at $N$ snapshots per orbit. We then fit linear probes for force, for the per-snapshot acceleration `accel_step` $\propto a^3/((1+m_2)\,r^2)$, for distance and for energy, and judge each probe against random-init controls. This draft rests on 12 replicated findings (7 supported, 5 refuted) from 28 evidence runs, 4 of which were cut short by the budget hard stop. Trained models reach `val_mse_ratio` of about 0.011 [f-20261011-201533-0653]. `accel_step` beats its control in every evidence run, and in dense-checkpoint runs it forms at steps 75–140 for $N = 16$, 64 and 256 alike. Whether it forms before the model beats the baseline depends on $N$, and the margins sit near the 0.1 bar [f-20261012-002713-8d93, f-20261012-005837-b9ac, f-20261012-015955-fc2b]. In those short runs, physical force mostly fails its control [f-20261012-005950-42a7, f-20261012-015955-fc2b]. The main negative result is that no $1/r^2$ law was extracted: symbolic fits to the probe directions give $r^{-1}$ forms such as $a^2/r - 0.55a$ in every seed tested [f-20261011-223027-8fb3, f-20261011-231118-7a37]. Fixing $m_2$ also did not lift force $R^2$ above 0.75 [f-20261011-223000-b5f9].

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

The lab has 12 citable findings, and all of them are replicated. Seven support their hypothesis and five refute it. Together they draw on 28 evidence runs. The budget hard stop cut four of those runs short. We point this out wherever their numbers appear.

### Prediction quality

Most trained models beat constant-velocity extrapolation by a wide margin.

- **Full runs:** final `val_mse_ratio` was about 0.011 at the default settings [f-20261011-201533-0653]. It was 0.0114–0.0119 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9], 0.0151–0.0157 in the transfer runs [f-20261011-203723-b7de], and 0.0128–0.0137 with $e \in [0, 0.9]$ [f-20261011-223027-8fb3]. The symbolic-replication runs reached 0.0115–0.0135 [f-20261011-231118-7a37].
- **Short dense-checkpoint runs:**
  - At $N = 64$, models ended at 0.43–0.53 [f-20261012-002713-8d93, f-20261012-005950-42a7].
  - At $N = 16$, they reached 0.017–0.018 [f-20261012-015955-fc2b].
  - At $N = 256$, the models never beat the baseline. They stayed 11.4–13.5× worse than constant-velocity extrapolation [f-20261012-005837-b9ac].
- **Runs cut short:**
  - Seed 14 stopped at completion 0.55 with `val_mse_ratio` 0.063. Seed 15 stopped at 0.917 with 0.0145 [f-20261011-223000-b5f9].
  - The $e \in [0, 0.8]$ runs stopped at completion 0.867 and 0.9, with 0.0178 and 0.0187 [f-20261011-211551-d718].

### When structure forms

**Supported: per-snapshot acceleration structure forms early in training, at a roughly fixed step across `steps_per_orbit`.** The step at which the `accel_step` margin first reaches its required 0.1 over control varies little across experiments:

| Setting | Checkpoint spacing / controls | `accel_step` formation step | `val_mse_ratio` first < 1 |
|---|---|---|---|
| $N = 64$, default E1 [f-20261011-201533-0653] | 250 steps, single control | 250 (both seeds; first trained checkpoint) | < 0.1 at step 1500 |
| $N = 64$ [f-20261011-203659-4e2f] | 25 steps, single control | 100 (both seeds) | first reported at step 175 |
| $N = 64$ [f-20261012-002713-8d93] | 10 steps, best-of-3 | 140, 140, 100 | 180, 180, 190 |
| $N = 64$ [f-20261012-005950-42a7] | best-of-5 | 85, 85, 90 | 135–165 |
| $N = 256$ [f-20261012-005837-b9ac] | 10 steps, best-of-5 | 100, 110, 120 | never |
| $N = 16$ [f-20261012-015955-fc2b] | best-of-5 | 90, 75, 80 | 20 (all seeds) |

- **$N = 256$:** it falls inside the predicted [90, 150] window, so the hypothesis that `accel_step` forms later at higher $N$ is refuted at the 0.1 threshold [f-20261012-005837-b9ac].
- **$N = 16$:** formation is 5, 15 and 5 steps from the seed-matched $N = 64$ runs (85, 90, 85) [f-20261012-015955-fc2b].

**"Forms before beating the baseline" depends on the baseline, not on when the structure forms.**

- At $N = 64$, `accel_step` crossed 0.1 before `val_mse_ratio` fell below 1. The lead was 40–90 steps with best-of-3 controls [f-20261012-002713-8d93] and 45–75 steps with best-of-5 controls [f-20261012-005950-42a7].
- At $N = 16$ the order flips. `val_mse_ratio` drops below 1 at step 20 in all three seeds (0.80, 0.77, 0.76). That is 55–70 steps before `accel_step` forms, while the formation step itself barely moves [f-20261012-015955-fc2b].
- At $N = 256$, `accel_step` forms in models that never beat the baseline [f-20261012-005837-b9ac].
- Taken together, "`val_mse_ratio` < 1" is a baseline-dependent marker and should not be used to time formation across $N$ [f-20261012-015955-fc2b].

![Formation trajectory of probe margins and val_mse_ratio, seed 49 (N = 16, best-of-5 controls)](figure:formation/run-20261012-015504-4f99)

![Formation trajectory of probe margins and val_mse_ratio, seed 46 (N = 64, 10-step checkpoints, best-of-3 controls)](figure:formation/run-20261011-234250-a7ca)

![Formation trajectory of probe margins and val_mse_ratio, seed 29 (N = 256, best-of-5 controls)](figure:formation/run-20261012-003504-b94e)

**Distance and energy form first.**

- In the dense runs, `distance` and `energy` crossed their bars at steps 5–40 [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-005837-b9ac, f-20261012-005950-42a7, f-20261012-015955-fc2b].
- In seed 45, `distance` beat its control at step 0, at initialisation [f-20261012-002713-8d93].
- This early signal is probably input geometry rather than learned dynamics [f-20261011-203659-4e2f].
- Random models differ by up to about 0.2 on these two targets, so margins measured against a single control should be read with caution.

**A transient force code appears early and then fades.**

- **The spike:** with best-of-5 controls, the force margin was:
  - 0.166–0.189 at step 10 at $N = 64$ [f-20261012-005950-42a7];
  - 0.127–0.242 at steps 10–20 at $N = 256$ [f-20261012-005837-b9ac];
  - 0.106–0.127 at step 10 at $N = 16$ [f-20261012-015955-fc2b].
- **The decay:**
  - At $N = 64$ it fell to about 0.00–0.02 by steps 40–50.
  - At $N = 256$ it fell below 0.1 by steps 30–60.
  - At $N = 16$ it fell to about 0 by steps 20–30.
- **Not a control artefact:** at $N = 64$ the spike comes from the trained model's force $R^2$ itself rising (from about 0.37–0.47 at step 0 to 0.61–0.64 at step 10) and then falling back, not from a low draw of controls. However, per-checkpoint control spread was not reported [f-20261012-005950-42a7].
- **Earlier runs:** force formed later than `accel_step` or never. It formed at step 750 vs 250 in seed 3, at step 450 vs 100 in seed 5, and never in seed 4 [f-20261011-201533-0653, f-20261011-203659-4e2f].

**Caveats on timing.**

- **Margins at the crossing are thin**, between 0.100 and 0.158 [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-005837-b9ac].
- **The `accel_step` margin levels off near the bar.**
  - At $N = 64$ with best-of-5 controls, the margin never reached 0.15 in two of three seeds [f-20261012-005950-42a7].
  - At $N = 16$ it never reached 0.15 in any seed (maximum 0.144) [f-20261012-015955-fc2b].
  - At $N = 256$, a 0.15 bar would push seeds 50 and 51 to steps 230–240 [f-20261012-005837-b9ac].
- **"Formation" is a threshold crossing on a smooth ramp, not a sharp event** [f-20261012-005837-b9ac].
- **Threshold-free measures keep the $N = 64$ ordering.** Half of the final margin is reached at steps 50–65. A headroom-normalised margin of at least 0.5 is first reached at steps 100, 125 and 130, which in seed 49 is only 5 steps before `val_mse_ratio` < 1 [f-20261012-005950-42a7].
- **A loss-based definition could reverse the order.** By step 50, `val_mse_ratio` had already dropped from about 70–80 to about 6–8 [f-20261012-002713-8d93].
- **Forming early is a correlation in time.** It does not show that the model uses the structure to predict [f-20261011-201533-0653].
- **Distance has not been ruled out.** Nobody has checked whether regressing `distance` out of the `accel_step` target removes the margin [f-20261012-002713-8d93].
- **One reporting inconsistency:** for $N = 16$ seed 48, the trajectory shows a margin of 0.102 at step 70, but `formed_at_step` is 75 [f-20261012-015955-fc2b].

### Force versus `accel_step`

Next-step prediction needs the per-snapshot acceleration $|a_{\text{step}}| = (2\pi/N)^2\, a^3/((1+m_2)\, r^2)$, not the physical force $|F| = m_2/r^2$.

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
| $N = 64$, 10-step, best-of-3 [f-20261012-002713-8d93] | 45 | 0.544 (0.504) | 0.040 | 0.884 (0.775) | 0.109 |
| | 44 | 0.190 (0.190) | 0.000 | 0.905 (0.786) | 0.119 |
| | 46 | 0.543 (0.482) | 0.062 | 0.898 (0.740) | 0.158 |
| $N = 64$, best-of-5 [f-20261012-005950-42a7] | 47 | 0.485 (0.447) | 0.039 | 0.904 (0.770) | 0.134 |
| | 49 | 0.486 (0.440) | 0.046 | 0.884 (0.750) | 0.134 |
| | 48 | 0.523 (0.464) | 0.059 | 0.892 (0.725) | 0.167 |
| $N = 256$, best-of-5 [f-20261012-005837-b9ac] | 29 | 0.183 (0.184) | −0.001 | 0.882 (0.705) | 0.177 |
| | 50 | 0.529 (0.473) | 0.056 | 0.904 (0.753) | 0.151 |
| | 51 | 0.546 (0.510) | 0.036 | 0.884 (0.733) | 0.151 |
| $N = 16$, best-of-5 [f-20261012-015955-fc2b] | 47 | 0.497 (0.467) | 0.030 | 0.909 (0.793) | 0.117 |
| | 48 | 0.543 (0.488) | 0.055 | 0.897 (0.752) | 0.144 |
| | 49 | 0.578 (0.456) | 0.123 | 0.907 (0.770) | 0.137 |

**Supported: in the short dense-checkpoint runs, the model holds a per-snapshot acceleration code but mostly no lasting physical-force code** [f-20261012-002713-8d93, f-20261012-005837-b9ac, f-20261012-005950-42a7, f-20261012-015955-fc2b].

- **`accel_step`** beats its control at the final checkpoint in every one of these runs.
- **Force** fails its required margin in every $N = 64$ and $N = 256$ dense run.
- **The exception:** at $N = 16$, force beats its control only in seed 49, where it formed at step 125 with final margin 0.123 [f-20261012-015955-fc2b].
- **Longer runs:** force keeps a margin of about 0.17–0.31 [f-20261011-201533-0653, f-20261011-203723-b7de, f-20261011-231118-7a37]. However, force and `accel_step` are collinear through $1/r^2$, so part of that signal may be distance information [f-20261011-203628-5cd9].

**Refuted: widening the mass range makes force fall behind `accel_step`** [f-20261011-203628-5cd9].

- **Result:** with $m_2 \in [0.01, 1]$, force kept the larger margin in both seeds.
- **The margin criterion favours force.** `accel_step`'s control $R^2$ of 0.71–0.75 leaves it little room to show a margin. Measured as a fraction of headroom, `accel_step` leads: 0.70 vs 0.48 in seed 2 and 0.71 vs 0.42 in seed 3.

**Refuted: fixing $m_2$ lifts the force probe** [f-20261011-223000-b5f9].

- **Prediction:** with $m_2 = 0.5$, force $R^2$ would reach 0.85.
- **Result:** force $R^2$ reached only 0.750 and 0.732, while `distance` was decoded at 0.87–0.89. The force ceiling therefore comes from the $1/r^2$ nonlinearity or how it is encoded, not from missing $m_2$ information.
- **Both runs were cut short by the budget hard stop.**
  - Seed 14 (completion 0.55) is below the lab's 80% bar and does not count as an agreeing seed.
  - Seed 15 (completion 0.917) was better trained and had the slightly lower force $R^2$.

**Where each quantity is read out.**

- **`accel_step`** is best at layer 4 in every $N = 64$ and $N = 16$ run, and at layer 3 in every $N = 256$ run [f-20261012-005837-b9ac, f-20261012-015955-fc2b].
- **`energy` and `distance`** are mostly best at layer 1, and at layer 2 in some runs [f-20261011-223027-8fb3, f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-015955-fc2b].
- **Force** has no stable best layer: it peaks at layer 0, 1, 2, 3 or 4 depending on the seed. It sits at layer 0 only where it equals its control [f-20261011-201533-0653, f-20261011-223000-b5f9, f-20261012-002713-8d93, f-20261012-005837-b9ac].

![Layer × checkpoint probe heatmap for force, seed 44 (force never beats control)](figure:heatmap/run-20261011-234250-6e25/force)

![Layer × checkpoint probe heatmap for accel_step, seed 44](figure:heatmap/run-20261011-234250-6e25/accel_step)

### Out-of-distribution transfer

**Refuted: training on restricted eccentricity transfers to high eccentricity within 10×** [f-20261011-203723-b7de].

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× the in-distribution value. The best force $R^2$ held at 0.75–0.77.
- **High eccentricity ($e \in [0.5, 0.8]$): failed.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution, and force $R^2$ fell to 0.40 and 0.34.
- **Every two-body ratio stayed below 1.**
- **Three-body ratios of 72 and 70 are exploratory only.**
- **OOD probes have no random controls.**

**Supported: training on $e \in [0, 0.8]$ removes the prediction gap** [f-20261011-211551-d718].

- **Gap closed:** held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution, well under the 3× threshold.
- **Early stop:** the budget hard stop cut both runs short at completion 0.867 and 0.9. Both are above the 80% bar.
- **Probe gap remained:** on high-$e$ systems the force probe reached only 0.366–0.377, against 0.729–0.802 on low-$e$ systems, even though prediction error was the same on both.

### Symbolic recovery of the law

**Refuted: wide eccentricity coverage yields $r^{-2}$ from the probe directions** [f-20261011-223027-8fb3].

- **`accel_step` fits ($e \in [0, 0.9]$):**
  - Seed 16: $0.0103\,a^2/r$, with error 0.71.
  - Seed 17: $0.00277\,\big((a+0.685)^3/r - r\big)$, with error 0.62.
- **Force fits:** $m_2/r + 0.40$ and $(m_2 + 0.60)/r$.
- **Weak probes:** `accel_step` $R^2$ was only 0.395 and 0.503, so the fits describe a weak linear approximation.

**Supported: the linearly decodable gravity direction has an $r^{-1}$ orbit-scale form** [f-20261011-231118-7a37].

- **Better probes, same result:** with `accel_step` $R^2$ of 0.831 and 0.881, the fits were still $r^{-1}$.
  - Seed 24: $a^2/r - 0.55a + 0.076r$ (scaled by 0.0134), with error 0.245.
  - Seed 26: $0.0136\,(a^2/r - 0.56a) + 0.0015$, with error 0.201.
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
  - Every $N = 64$ and $N = 256$ dense run [f-20261012-002713-8d93, f-20261012-005837-b9ac, f-20261012-005950-42a7].
  - Two of three $N = 16$ runs [f-20261012-015955-fc2b].
- **Linear decodability and prediction come apart:**
  - At high eccentricity [f-20261011-211551-d718, f-20261011-223027-8fb3].
  - At $N = 256$, where `accel_step` forms in models that never beat the baseline [f-20261012-005837-b9ac].
  - At $N = 16$, where the model beats the baseline well before `accel_step` forms [f-20261012-015955-fc2b].
- **Not yet reported:** `velocity` probe results and activation-patching results.
- **Limits throughout:**
  - All probes are linear.
  - Formation steps depend on the 0.1 threshold, which sits close to the `accel_step` ceiling because its control $R^2$ is high.
  - The budget hard stop cut four evidence runs short. One of them (seed 14, completion 0.55) is below the replication bar [f-20261011-211551-d718, f-20261011-223000-b5f9].
