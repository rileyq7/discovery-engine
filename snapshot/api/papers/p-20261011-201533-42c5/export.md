# Discovery Engine: running draft

## Abstract

Do small transformers trained to predict simulated two-body orbits encode the gravitational law, and when does that structure form? We probe the residual stream with linear probes for the physical force $m_2/r^2$, the per-snapshot acceleration `accel_step` $\propto a^3/((1+m_2)r^2)$, distance and energy. A probe result counts only if its $R^2$ beats a random-init control by at least 0.1. The six findings were all replicated across two seeds: three were supported and three refuted. They rest on 12 runs, two of which were cut short by the budget stop. The models beat constant-velocity extrapolation, with a final `val_mse_ratio` of about 0.011 [f-20261011-201533-0653]. `accel_step` structure formed before prediction beat the baseline: its margin reached 0.1 at step 100, while `val_mse_ratio` was still 1.96 and 1.84 [f-20261011-203659-4e2f]. Error on high-eccentricity systems was 16–18× the in-distribution error [f-20261011-203723-b7de]. That gap closed to about 1.0× once training covered $e \in [0, 0.8]$, in runs stopped at 0.867 and 0.9 completion [f-20261011-211551-d718]. Widening $m_2$ did not make the force margin fall behind `accel_step` [f-20261011-203628-5cd9]. The main negative result is that symbolic fits to the probe directions gave $r^{-1}$, not $r^{-2}$ (fit errors 0.71 and 0.62), so no explicit inverse-square law has been extracted [f-20261011-223027-8fb3].

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

The lab has six citable findings, all replicated across two seeds: three supported and three refuted. They rest on 12 evidence runs. The budget hard stop cut 2 of those runs short. All models share one architecture ($L = 4$, $d = 64$, $N = 64$) [f-20261011-211551-d718, f-20261011-223027-8fb3]. Each probe result below counts only if its $R^2$ beats a random-init control of the same architecture by at least 0.1.

### Prediction quality

Every trained model beat constant-velocity extrapolation on held-out two-body systems.

- **Final in-distribution `val_mse_ratio`:**
  - About 0.011 in the default-range runs [f-20261011-201533-0653].
  - 0.0114 and 0.0119 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9].
  - 0.0151 and 0.0157 in the restricted-eccentricity transfer runs [f-20261011-203723-b7de].
  - 0.0178 and 0.0187 with $e \in [0, 0.8]$ [f-20261011-211551-d718].
  - 0.0137 and 0.0128 with $e \in [0, 0.9]$ [f-20261011-223027-8fb3].
- **The dense-checkpoint runs ended higher,** at 0.141 and 0.166 [f-20261011-203659-4e2f].

### When structure forms

**Supported: gravity-related structure forms before the model beats the baseline** [f-20261011-201533-0653, f-20261011-203659-4e2f].

- **250-step checkpoints (seeds 2 and 3)** [f-20261011-201533-0653]:
  - `accel_step` already beat its control at the first checkpoint, step 250. Its margins were 0.158 and 0.127 at layer 4.
  - `distance` margins at the same checkpoint were 0.597 and 0.681 at layer 1.
  - `val_mse_ratio` first fell below 0.1 only at step 1500 (0.0858 and 0.0985).
- **25-step checkpoints (seeds 4 and 5)** [f-20261011-203659-4e2f]:
  - In both seeds the `accel_step` margin first reached $\geq 0.1$ at step 100 (0.115 and 0.108, layer 4).
  - At that point `val_mse_ratio` was still 1.96 and 1.84. It was first reported below 1 at step 175 (0.736 and 0.878).
  - `distance` and `energy` formed at step 25, while `val_mse_ratio` was still between 5 and 83.

![Formation trajectory of probe margins and val_mse_ratio, seed 2 (250-step checkpoints)](figure:formation/run-20261011-195457-2ec2)

![Formation trajectory of probe margins and val_mse_ratio, seed 4 (25-step checkpoints)](figure:formation/run-20261011-202258-3cf9)

**Force never formed before `accel_step`.**

- Seed 2: both formed at step 250. Seed 3: force formed at step 750, `accel_step` at 250 [f-20261011-201533-0653].
- Seed 5: force formed at step 450, `accel_step` at 100. Seed 4: force never formed, with a final margin of 0.068 [f-20261011-203659-4e2f].

**Caveats on timing.**

- The `accel_step` margins at the crossing are only just above the threshold (0.108–0.158). With a 0.12 threshold, the dense-run crossing would move to about step 175 and coincide with `val_mse_ratio` < 1 rather than come before it [f-20261011-203659-4e2f].
- Seed 5's `val_mse_ratio` rose back to 1.057 at step 225, so "first beats baseline" is not a stable transition [f-20261011-203659-4e2f].
- Both trajectories skip some checkpoints [f-20261011-201533-0653, f-20261011-203659-4e2f].
- Early `distance` and `energy` signal may be geometry that is easy to read from the input embedding, not learned dynamics [f-20261011-203659-4e2f].
- Forming early is a correlation in time. It does not show that the model uses this structure to predict [f-20261011-201533-0653].

### Force versus `accel_step`

The physical force is $|F| = m_2/r^2$. Next-step prediction needs the per-snapshot acceleration, $|a_{\text{step}}| = (2\pi/N)^2\, a^3/((1+m_2)\, r^2)$.

- **Raw $R^2$:** in every evidence run `accel_step` has the higher raw $R^2$ (0.395–0.935, against 0.154–0.738 for force).
- **Margin over control:** which probe leads depends on the configuration (table below).

| Experiment | Seed | force $R^2$ (control) | force margin | `accel_step` $R^2$ (control) | `accel_step` margin |
|---|---|---|---|---|---|
| Default range [f-20261011-201533-0653] | 2 | 0.738 (0.516) | 0.223 | 0.914 (0.710) | 0.204 |
| | 3 | 0.696 (0.465) | 0.230 | 0.925 (0.752) | 0.172 |
| $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9] | 2 | 0.737 (0.494) | 0.243 | 0.914 (0.709) | 0.205 |
| | 3 | 0.681 (0.446) | 0.235 | 0.928 (0.753) | 0.175 |
| Dense checkpoints [f-20261011-203659-4e2f] | 4 | 0.574 (0.505) | 0.068 | 0.898 (0.742) | 0.156 |
| | 5 | 0.624 (0.521) | 0.103 | 0.904 (0.740) | 0.165 |
| Transfer runs, in-distribution [f-20261011-203723-b7de] | 7 | 0.738 (0.529) | 0.209 | 0.934 (0.757) | 0.178 |
| | 6 | 0.689 (0.523) | 0.166 | 0.935 (0.727) | 0.208 |
| $e \in [0, 0.8]$, stopped early [f-20261011-211551-d718] | 9 | 0.452 (0.244) | 0.208 | 0.715 (0.378) | 0.337 |
| | 8 | 0.528 (0.216) | 0.312 | 0.707 (0.349) | 0.358 |
| $e \in [0, 0.9]$, $m_2 \in [0.01, 1]$ [f-20261011-223027-8fb3] | 16 | 0.154 (0.057) | 0.097 | 0.395 (0.175) | 0.220 |
| | 17 | 0.237 (0.131) | 0.106 | 0.503 (0.182) | 0.321 |

**Refuted: widening the mass range does not make force fall behind `accel_step`** [f-20261011-203628-5cd9].

- **Prediction:** with $m_2 \in [0.01, 1]$, the force probe would fall behind `accel_step`.
- **Result:** force kept the larger margin in both seeds (0.243 and 0.235, against 0.205 and 0.175). Its margins were no lower than in the default-range runs (0.223–0.230).
- **The criterion is biased against `accel_step`.** Its control $R^2$ is already 0.71–0.75, which leaves little room for a margin. Measured as a fraction of available headroom, $(R^2 - R^2_{\text{control}})/(1 - R^2_{\text{control}})$, `accel_step` leads: 0.70 vs 0.48 in seed 2 and 0.71 vs 0.42 in seed 3.
- **What this shows:** some force-specific signal exists beyond the control. It does not show that force is the primary representation [f-20261011-203628-5cd9].

![Layer × checkpoint probe heatmap for force, seed 4 (force never beats control)](figure:heatmap/run-20261011-202258-3cf9/force)

![Layer × checkpoint probe heatmap for accel_step, seed 4](figure:heatmap/run-20261011-202258-3cf9/accel_step)

**Where each quantity is read out.**

- `accel_step` is best at layer 4 in every evidence run.
- `energy` is best at layer 1 in every evidence run.
- `distance` is best at layer 1 everywhere except seed 16, where it is best at layer 2 [f-20261011-223027-8fb3].
- Force has no stable best layer. It is best at layer 4 in some seeds and layer 1 in others [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f, f-20261011-203723-b7de, f-20261011-211551-d718, f-20261011-223027-8fb3].
- Force and `accel_step` are collinear through $1/r^2$, and `distance` margins are large (0.64–0.70 in the wide-mass runs). Part of the force signal may therefore be distance information [f-20261011-203628-5cd9].

### Out-of-distribution transfer

**Refuted: training on restricted eccentricity transfers to high eccentricity** [f-20261011-203723-b7de]. In seeds 7 and 6, in-distribution `val_mse_ratio` was 0.0151 and 0.0157.

- **Low mass ratio ($m_2 \in [0.01, 0.1]$): passed.** `val_mse_ratio` was 0.057 and 0.055, about 3.6× in-distribution. Force $R^2$ held at 0.75–0.77.
- **High eccentricity ($e \in [0.5, 0.8]$): failed the pre-registered 10× criterion.** `val_mse_ratio` was 0.270 and 0.249, about 16–18× in-distribution. Force $R^2$ fell to 0.40 and 0.34, from 0.74 and 0.69.
- **All two-body ratios stayed below 1.**
- **Three-body ratios of 72 and 70 are exploratory only.**
- **OOD probe values have no random controls,** so the force $R^2$ drop cannot be read as a margin.

**Supported: covering $e \in [0, 0.8]$ in training removes the high-eccentricity gap** [f-20261011-211551-d718].

- **Early stop:** both runs (seeds 9 and 8) were cut short by the budget hard stop, at completion 0.867 and 0.9. Their probes describe the weights at the stop, and `val_mse_ratio` was still falling. Both runs are above the 80% completion bar, so they count toward replication.
- **Prediction gap closed:** held-out high-$e$ `val_mse_ratio` was 0.0187 and 0.0189, about 1.05× and 1.01× in-distribution. The pre-registered threshold was 3×. Held-out low-$e$ systems scored 0.0185 and 0.0215, so high $e$ was no worse.
- **Probes still beat their controls** (table above).
- **Probe gap remained:** the best force probe on held-out high-$e$ systems reached only 0.366–0.377, against 0.729–0.802 on low-$e$ systems, even though prediction error was the same on both [f-20261011-211551-d718].

### Symbolic recovery of the law

**Refuted: wide eccentricity coverage yields an $r^{-2}$ law from the probe directions** [f-20261011-223027-8fb3]. Seeds 16 and 17 were trained with $e \in [0, 0.9]$ and $m_2 \in [0.01, 1]$. Symbolic regression on the best `accel_step` direction gave $r^{-1}$ in both seeds:

- **Seed 16:** $0.0103\,a^2/r$, with error against the true law 0.71.
- **Seed 17:** $0.00277\,\big((a+0.685)^3/r - r\big)$. This is roughly $a^3$ but still $r^{-1}$, with error 0.62.
- **Force fits:** $m_2/r + 0.40$ and $(m_2 + 0.60)/r$, with errors 0.90 and 0.64.
- **No fit recovered the law.**

**Probe quality fell while prediction held.**

- `val_mse_ratio` stayed at 0.0137 and 0.0128.
- `accel_step` raw $R^2$ fell to 0.395 and 0.503, from 0.92–0.95 at $e \leq 0.5$. It still beat its control, with margins 0.220 and 0.321.
- The force margin was 0.097 in seed 16, below the 0.1 threshold, and 0.106 in seed 17.
- The $r^{-1}$ result may come from a linear probe failing to follow the heavy-tailed $1/r^2$ target near pericentre, not from the model encoding an $r^{-1}$ quantity.
- These are the only evidence runs with symbolic fits [f-20261011-223027-8fb3].

![Layer × checkpoint probe heatmap for accel_step, seed 17 (e ∈ [0, 0.9])](figure:heatmap/run-20261011-215313-2e1f/accel_step)

### Summary of null and negative results

- **Three hypotheses were refuted:**
  - Widening $m_2$ did not make force fall behind `accel_step` [f-20261011-203628-5cd9].
  - Restricted-$e$ training did not transfer to high $e$ within 10× [f-20261011-203723-b7de]. A follow-up traced this to training coverage [f-20261011-211551-d718].
  - Symbolic fits gave $r^{-1}$, not $r^{-2}$ [f-20261011-223027-8fb3].
- **No explicit $1/r^2$ law has been extracted** [f-20261011-223027-8fb3].
- **Force sometimes never beats its control** (seeds 4 and 16) [f-20261011-203659-4e2f, f-20261011-223027-8fb3].
- **Linear decodability and prediction come apart at high eccentricity** [f-20261011-211551-d718, f-20261011-223027-8fb3].
- **Not yet reported:** no `velocity` probe results and no activation-patching results.
- **Limits that apply throughout:** all evidence uses linear probes, two seeds per finding and one architecture.
