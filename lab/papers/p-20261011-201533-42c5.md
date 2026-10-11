# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated two-body orbits (REBOUND, $G = M = 1$) learn the gravitational law or heuristics that happen to work, and when any such structure forms. Linear probes read the residual stream for force $m_2/r^2$, per-snapshot acceleration (`accel_step`) $(2\pi/N)^2 a^3/((1+m_2)r^2)$, distance and energy. Each probe is compared against a randomly initialised control, and symbolic regression is fitted to the probe directions. All 9 findings replicated (4 supported, 5 refuted) across 18 evidence runs. The budget hard stop cut 4 of those runs short. Models beat constant-velocity extrapolation, reaching `val_mse_ratio` of about 0.011 [f-20261011-201533-0653]. `accel_step` beats its control by step 100, while `val_mse_ratio` is still 1.96 and 1.84 [f-20261011-203659-4e2f]. Force forms later or never, and the timing did not shift at $N = 256$ [f-20261011-231210-478d]. Restricted-eccentricity training left a 16–18× high-$e$ gap [f-20261011-203723-b7de]. Training on $e \in [0, 0.8]$ closed it, in runs stopped early at completion 0.867 and 0.9 [f-20261011-211551-d718]. The main negative result is that no $1/r^2$ law was recovered. All four symbolic fits of `accel_step` give $r^{-1}$ forms [f-20261011-223027-8fb3, f-20261011-231118-7a37].

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

The lab has nine citable findings. All nine are replicated across two seeds: four are supported and five are refuted. They rest on 18 evidence runs, and the budget hard stop cut 4 of those runs short. A probe result counts only when its $R^2$ beats a random-init control of the same architecture by at least 0.1. Most evidence runs use $L = 4$, $d = 64$ and $N = 64$ snapshots per orbit [f-20261011-211551-d718]. One experiment raised $N$ to 256 [f-20261011-231210-478d].

### Prediction quality

Almost every trained model beat constant-velocity extrapolation on held-out two-body systems. The exception is the $N = 256$ runs.

- **Final in-distribution `val_mse_ratio`:**
  - About 0.011 in the default-range runs [f-20261011-201533-0653].
  - 0.0114 and 0.0119 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9].
  - 0.0151 and 0.0157 in the restricted-eccentricity transfer runs [f-20261011-203723-b7de].
  - 0.0178 and 0.0187 with $e \in [0, 0.8]$. Both runs were stopped early, at completion 0.867 and 0.9 [f-20261011-211551-d718].
  - 0.0137 and 0.0128 with $e \in [0, 0.9]$ [f-20261011-223027-8fb3].
  - 0.0135 and 0.0115 in the symbolic-replication runs (seeds 24 and 26) [f-20261011-231118-7a37].
  - 0.063 and 0.0145 with $m_2$ fixed at 0.5. Both runs were stopped early: seed 14 at completion 0.55 and seed 15 at 0.917 [f-20261011-223000-b5f9].
- **The dense-checkpoint runs ended higher,** at 0.141 and 0.166 [f-20261011-203659-4e2f].
- **The $N = 256$ runs did not beat the baseline.** Their `val_mse_ratio` was 3.12–3.20 after all 600 planned steps [f-20261011-231210-478d].

### When structure forms

**Supported: gravity-related structure forms before the model beats the baseline** [f-20261011-201533-0653, f-20261011-203659-4e2f].

- **250-step checkpoints (seeds 2 and 3)** [f-20261011-201533-0653]:
  - `accel_step` already beat its control at step 250, with margins of 0.158 and 0.127 at layer 4.
  - `distance` margins at step 250 were 0.597 and 0.681.
  - `val_mse_ratio` first fell below 0.1 only at step 1500.
- **25-step checkpoints (seeds 4 and 5)** [f-20261011-203659-4e2f]:
  - The `accel_step` margin first reached $\geq 0.1$ at step 100 (0.115 and 0.108). At that step `val_mse_ratio` was still 1.96 and 1.84.
  - `val_mse_ratio` was first reported below 1 at step 175 (0.736 and 0.878).
  - `distance` and `energy` formed at step 25, while `val_mse_ratio` was still between 5 and 83.

**Refuted: formation timing does not scale with steps per orbit** [f-20261011-231210-478d].

- At $N = 256$ the `accel_step` margin first reached $\geq 0.1$ at step 100 in both seeds (0.116 and 0.104). At step 50 it had been 0.045–0.049. Step 100 is the same crossing step as at $N = 64$.
- `distance` and `energy` again formed by step 25.
- Force never beat its control. Its final margin was 0.012–0.032, after a brief peak of 0.09 at step 50.
- These models never learned the task (`val_mse_ratio` 3.12–3.20). That makes it hard to read the timing as set by task demand.

![Formation trajectory of probe margins and val_mse_ratio, seed 4 (N = 64, 25-step checkpoints)](figure:formation/run-20261011-202258-3cf9)

![Formation trajectory of probe margins and val_mse_ratio, seed 30 (N = 256)](figure:formation/run-20261011-230710-37e6)

**Force never formed before `accel_step`.**

- Seed 3: force formed at step 750, `accel_step` at step 250. Seed 2: both formed at step 250 [f-20261011-201533-0653].
- Seed 5: force formed at step 450, `accel_step` at step 100. Seed 4: force never formed [f-20261011-203659-4e2f].
- At $N = 256$, force never formed in either seed [f-20261011-231210-478d].

**Caveats on timing.**

- The crossing margins sit just above the threshold, at 0.104–0.158. With a 0.12 threshold, the dense-run crossing would move to about step 175 and coincide with `val_mse_ratio` falling below 1 [f-20261011-203659-4e2f, f-20261011-231210-478d].
- Seed 5's `val_mse_ratio` rose back to 1.057 at step 225 [f-20261011-203659-4e2f].
- Early `distance` and `energy` signal may be input geometry rather than learned dynamics [f-20261011-203659-4e2f].
- Forming early is a correlation in time. It does not show that the model uses this structure to predict [f-20261011-201533-0653].

### Force versus `accel_step`

The physical force is $|F| = m_2/r^2$. Next-step prediction needs the per-snapshot acceleration, $|a_{\text{step}}| = (2\pi/N)^2\, a^3/((1+m_2)\, r^2)$.

**Raw $R^2$.** In every evidence run, `accel_step` has the higher raw $R^2$: 0.395–0.946, against 0.154–0.750 for force.

**Margin over control.** Which probe leads on margin depends on the configuration:

| Experiment | Seed | force $R^2$ (control) | force margin | `accel_step` $R^2$ (control) | `accel_step` margin |
|---|---|---|---|---|---|
| Default range [f-20261011-201533-0653] | 2 | 0.738 (0.516) | 0.223 | 0.914 (0.710) | 0.204 |
| | 3 | 0.696 (0.465) | 0.230 | 0.925 (0.752) | 0.172 |
| $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9] | 2 | 0.737 (0.494) | 0.243 | 0.914 (0.709) | 0.205 |
| | 3 | 0.681 (0.446) | 0.235 | 0.928 (0.753) | 0.175 |
| Dense checkpoints [f-20261011-203659-4e2f] | 4 | 0.574 (0.505) | 0.068 | 0.898 (0.742) | 0.156 |
| | 5 | 0.624 (0.521) | 0.103 | 0.904 (0.740) | 0.165 |
| Transfer, in-distribution [f-20261011-203723-b7de] | 7 | 0.738 (0.529) | 0.209 | 0.934 (0.757) | 0.178 |
| | 6 | 0.689 (0.523) | 0.166 | 0.935 (0.727) | 0.208 |
| $e \in [0, 0.8]$, stopped early [f-20261011-211551-d718] | 9 | 0.452 (0.244) | 0.208 | 0.715 (0.378) | 0.337 |
| | 8 | 0.528 (0.216) | 0.312 | 0.707 (0.349) | 0.358 |
| $m_2 = 0.5$ fixed, stopped early [f-20261011-223000-b5f9] | 14 | 0.750 (0.560) | 0.19 | 0.924 (0.743) | 0.18 |
| | 15 | 0.732 (0.508) | 0.224 | 0.946 (0.762) | 0.18 |
| $e \in [0, 0.9]$, $m_2 \in [0.01, 1]$ [f-20261011-223027-8fb3] | 16 | 0.154 (0.057) | 0.097 | 0.395 (0.175) | 0.220 |
| | 17 | 0.237 (0.131) | 0.106 | 0.503 (0.182) | 0.321 |
| Symbolic replication [f-20261011-231118-7a37] | 24 | 0.692 (0.451) | 0.241 | 0.831 (0.487) | 0.344 |
| | 26 | 0.660 (0.395) | 0.265 | 0.881 (0.581) | 0.299 |
| $N = 256$ [f-20261011-231210-478d] | 30 | 0.542 (0.510) | 0.032 | 0.881 (0.672) | 0.209 |
| | 29 | 0.555 (0.543) | 0.012 | 0.890 (0.699) | 0.191 |

**Refuted: widening the mass range makes force fall behind `accel_step`** [f-20261011-203628-5cd9].

- **Result:** with $m_2 \in [0.01, 1]$, force kept the larger margin in both seeds.
- **The criterion is biased.** `accel_step`'s control $R^2$ of 0.71–0.75 leaves it little room for a margin. Measured as a fraction of the available headroom, $(R^2 - R^2_{\text{control}})/(1 - R^2_{\text{control}})$, `accel_step` leads: 0.70 vs 0.48 and 0.71 vs 0.42.

**Refuted: fixing $m_2$ lifts the force probe** [f-20261011-223000-b5f9].

- **Setup:** with $m_2 = 0.5$, the force is exactly $0.5/r^2$.
- **Prediction:** force $R^2$ would reach 0.85.
- **Result:** force $R^2$ reached only 0.750 and 0.732, while `distance` was decoded at 0.87–0.89. The force ceiling therefore comes from the $1/r^2$ nonlinearity or how the layers encode it, not from missing $m_2$ information.
- **Both runs were cut short by the budget hard stop.** Seed 15 reached completion 0.917. Seed 14 reached only 0.55 and stopped at step 1500, with `val_mse_ratio` 0.063. That is below the lab's 80% completion bar, so under the protocol seed 14 does not count as an agreeing seed. Treat this finding's replication with that in mind. The better-trained seed 15 had the slightly lower force $R^2$.

**Where each quantity is read out.**

- `energy` is best at layer 1 in every evidence run.
- `distance` is best at layer 1 in every run except seed 16, where it is best at layer 2 [f-20261011-223027-8fb3].
- `accel_step` is best at layer 4 in every run except the $N = 256$ runs, where it is best at layer 3 [f-20261011-231210-478d].
- Force has no stable best layer: layer 1, 2 or 4 depending on seed [f-20261011-201533-0653, f-20261011-223000-b5f9].
- Force and `accel_step` are collinear through $1/r^2$, so part of the force signal may be distance information [f-20261011-203628-5cd9].

![Layer × checkpoint probe heatmap for force, seed 4 (force never beats control)](figure:heatmap/run-20261011-202258-3cf9/force)

![Layer × checkpoint probe heatmap for accel_step, seed 4](figure:heatmap/run-20261011-202258-3cf9/accel_step)

### Out-of-distribution transfer

**Refuted: training on restricted eccentricity transfers to high eccentricity** [f-20261011-203723-b7de].

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× in-distribution.
- **High eccentricity ($e \in [0.5, 0.8]$): failed the pre-registered 10× criterion.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution. Force $R^2$ fell to 0.40 and 0.34.
- **All two-body ratios stayed below 1.**
- **Three-body ratios of 72 and 70 are exploratory only.**
- **OOD probes have no random controls.**

**Supported: covering $e \in [0, 0.8]$ in training removes the gap** [f-20261011-211551-d718].

- **Prediction gap closed:** held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution. The pre-registered threshold was 3×.
- **Early stop:** both runs were cut short by the budget hard stop at step 2500 of 3000 (completion 0.867 and 0.9), while `val_mse_ratio` was still falling. Both are above the 80% bar.
- **Probe gap remained:** the best force probe on high-$e$ systems reached only 0.366–0.377, against 0.729–0.802 on low-$e$ systems, even though prediction error was equal on both.

### Symbolic recovery of the law

**Refuted: wide eccentricity coverage yields $r^{-2}$ from the probe directions** [f-20261011-223027-8fb3].

- **Setup:** $e \in [0, 0.9]$, $m_2 \in [0.01, 1]$.
- **`accel_step` fits:**
  - Seed 16: $0.0103\,a^2/r$, with error 0.71.
  - Seed 17: $0.00277\,\big((a+0.685)^3/r - r\big)$, with error 0.62.
- **Force fits:** $m_2/r + 0.40$ and $(m_2 + 0.60)/r$.
- **Weak probes:** `accel_step` raw $R^2$ was only 0.395 and 0.503, so these fits describe a weak linear approximation.

**Supported: the linearly decodable gravity direction has an $r^{-1}$ orbit-scale form** [f-20261011-231118-7a37].

- **Better probes, same result:** seeds 24 and 26 reached `accel_step` $R^2$ of 0.831 and 0.881, and their fits were still $r^{-1}$.
  - Seed 24: $0.0134\,r\,(0.275 - a/r)^2$, which expands to $a^2/r - 0.55a + 0.076r$, with error 0.245.
  - Seed 26: $0.0136\,a\,(a/r - 0.562) + 0.0015$, with error 0.201.
- **Force fits** were $0.81/r$ and $0.80/r$, with errors 0.43–0.46.
- **Agreement:** together with an earlier seed, the pattern holds in 3 of 3 seeds.
- **Strong codes alongside:** `distance` ($R^2$ 0.83–0.85) and `energy` ($R^2$ 0.92–0.93) at layer 1.
- **Caveats:**
  - Seed 26 only just clears the fit-error criterion.
  - The fitted shape $a^2/r - c\,a$ with $c \approx 0.55$ is close to the vis-viva combination $(v\,dt)^2/a$, but this has not been established.
  - The symbolic-regression pipeline has not yet been checked on the true `accel_step` target. If it also returns $r^{-1}$ there, the result describes the fitter, not the model.

![Layer × checkpoint probe heatmap for accel_step, seed 24](figure:heatmap/run-20261011-225359-5030/accel_step)

### Summary of null and negative results

- **Five hypotheses were refuted:**
  - Wider $m_2$ did not make force fall behind `accel_step` [f-20261011-203628-5cd9].
  - Restricted-$e$ training did not transfer to high $e$ within 10× [f-20261011-203723-b7de]. This was later traced to training coverage [f-20261011-211551-d718].
  - Fixing $m_2$ did not lift force $R^2$ [f-20261011-223000-b5f9].
  - Wide-$e$ training did not yield an $r^{-2}$ fit [f-20261011-223027-8fb3].
  - Formation timing did not shift with $N = 256$ [f-20261011-231210-478d].
- **No explicit $1/r^2$ law has been extracted.** All four runs with symbolic fits give $r^{-1}$ [f-20261011-223027-8fb3, f-20261011-231118-7a37].
- **Force sometimes never beats its control:** seeds 4, 16, 29 and 30 [f-20261011-203659-4e2f, f-20261011-223027-8fb3, f-20261011-231210-478d].
- **Linear decodability and prediction come apart at high eccentricity** [f-20261011-211551-d718, f-20261011-223027-8fb3].
- **Not yet reported:** no `velocity` probe results and no activation-patching results.
- **Limits throughout:**
  - All probes are linear, with two seeds per finding.
  - Four evidence runs were cut short by the budget hard stop. One of them (seed 14, completion 0.55) is below the replication bar [f-20261011-211551-d718, f-20261011-223000-b5f9].
