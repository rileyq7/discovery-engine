# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated orbits ($G = M = 1$) encode the physical law $|F| = m_2/r^2$ or a substitute that works only for the task, and when any such structure forms. We fit linear probes to the residual stream and compare each against best-of-several random-model controls. The probes target both physical force and the per-snapshot acceleration $(F/m_2)\,dt^2$ (`accel_step`), which is what next-step prediction actually needs. We report 31 replicated findings (21 supported, 10 refuted) from 76 evidence runs; the budget hard stop cut 4 of those runs short. At $N = 64$, `accel_step` clears the 0.1 margin at steps 85–140, before `val_mse_ratio` falls below 1 [f-20261012-002713-8d93, f-20261012-005950-42a7]. It leads force on headroom-normalised margin in all multi-seed comparisons but one, for example 0.725/0.740 against 0.44/0.45 [f-20261013-054512-1b08]. Force forms later and peaks at the `distance` layer [f-20261013-020028-8792]. It also weakens with denser sampling: on data seed 1 the margin is 0.209 at $N = 64$ and 0.137/0.145 at $N = 128$ [f-20261013-054542-23e1]. The main negative result is that symbolic fits to the probe directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3].

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

The lab has 31 citable findings, all replicated. Of these, 21 support their hypothesis and 10 refute it. They rest on 76 evidence runs, and the budget hard stop cut 4 of those runs short. Every probe result below is a margin over the best of several randomly initialised controls, unless a finding says otherwise.

### Prediction quality

The trained models beat constant-velocity extrapolation by a wide margin:

- The seeded E1 setup ended at a `val_mse_ratio` of about 0.011 [f-20261011-201533-0653].
- At $N = 16$ with 3000 steps it reached 0.0007 [f-20261013-000329-a86b].
- The 6-layer, $d_{\text{model}} = 128$ model reached 0.0037 and 0.0032 [f-20261012-030233-ab28].
- At $N = 64$ with 12000 steps it reached 0.0030 and 0.0031 [f-20261013-051738-2a37].

The one exception is dense sampling with short training. At $N = 256$, $e \le 0.85$ and 600 steps, the models never beat the baseline: final `val_mse_ratio` was 2.03–2.37 [f-20261012-034134-5e82]. At $N = 256$ with 3000 steps they did, at 0.237 and 0.282 [f-20261013-054603-d6cc].

### Per-snapshot acceleration forms early and at a stable training step

**`accel_step` is linearly decodable above its control in every configuration tested.** It forms early, usually before the model beats the baseline:

| Setting | `accel_step` forms (step) | `val_mse_ratio` first < 1 (step) |
|---|---|---|
| E1, 250-step checkpoints | 250, 250 | 1500, 1500 [f-20261011-201533-0653] |
| $N = 64$, 25-step checkpoints | 100, 100 | 175 [f-20261011-203659-4e2f] |
| $N = 64$, 10-step checkpoints | 140, 140, 100 | 180, 180, 190 [f-20261012-002713-8d93] |
| $N = 64$, best-of-5 controls | 85–90 | 135–165 [f-20261012-005950-42a7] |
| $N = 64$, seeds 62–64 | 90, 100, 100 | 140–160 [f-20261012-044142-d490] |
| $N = 64$, 5-step checkpoints, 5 seeds | 65–90 | — [f-20261012-054929-0067] |

`distance` and `energy` form earlier still, around steps 5–40, while the model is far worse than the baseline [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-005950-42a7].

**The formation step depends on training steps, not on loss level.** Several results point the same way:

- **$N = 16$.** `val_mse_ratio` falls below 1 at step 20, but `accel_step` forms at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs. The ordering flips because the baseline is weaker at coarse sampling; the formation step barely moves [f-20261012-015955-fc2b].
- **$N = 256$ at $e \le 0.85$.** `accel_step` forms at steps 60–70 [f-20261012-034134-5e82], the same window as the paired $N = 64$ arm (60, 60, 70) [f-20261012-034018-5526], even though these models never beat the baseline.
- **Learning rate.** Halving it delays formation only to steps 70, 80 and 100 [f-20261012-044258-ab88].

![Formation timeline, N = 16, seed 47: val_mse_ratio drops below 1 before accel_step forms](figure:formation/run-20261012-015504-ae8c)

![accel_step probe heatmap, N = 64, seed 62](figure:heatmap/run-20261012-043022-3319/accel_step)

**Caveat on thin margins.** At $e \le 0.5$ the `accel_step` control is high (0.71–0.79), so final margins sit close to the 0.1 bar: 0.109–0.158 [f-20261012-002713-8d93]. In two of three seeds the margin never reached 0.15 [f-20261012-005950-42a7]. Formation steps therefore depend on the threshold.

### Physical force: absent in short runs, late and growing in long ones

**In short runs, force mostly did not beat its control.** Final force margins were:

- 0.040, 0.000 and 0.062 [f-20261012-002713-8d93];
- 0.039–0.059 [f-20261012-005950-42a7];
- 0.033–0.080 [f-20261012-044142-d490];
- 0.065–0.078 at half the learning rate [f-20261012-044258-ab88];
- 0.004–0.031 at $N = 256$ [f-20261012-034134-5e82].

There were a few borderline exceptions: 0.103 and 0.105 in 2 of 5 seeds [f-20261012-054929-0067], and 0.123 in one $N = 16$ seed [f-20261012-015955-fc2b].

**Every short run also showed a brief early force spike.** The margin peaked at about 0.11–0.19 around steps 10–25, then fell to about 0 by steps 40–50 [f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-054929-0067]. Best-of-5 controls did not remove it [f-20261012-005950-42a7].

**With longer training, a force code appears.** Final force margins after 1500–12000 steps:

- 0.223 and 0.230 [f-20261011-201533-0653];
- 0.243 and 0.235 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9];
- 0.247 and 0.234 at $N = 64$, 3000 steps [f-20261013-015914-3f49];
- 0.209 and 0.209 at $N = 64$, 3000 steps, data seed 1 [f-20261013-054512-1b08];
- 0.255 and 0.266 at $N = 64$, 12000 steps, gaining only 0.019 and 0.007 between steps 6000 and 12000 [f-20261013-051738-2a37];
- 0.236 and 0.275 at $N = 16$ [f-20261013-000329-a86b];
- 0.221 and 0.255 in a 2-layer, $d = 32$ model [f-20261013-000220-ab2d];
- 0.300 and 0.333 in the 6-layer, $d_{\text{model}} = 128$ model [f-20261012-030233-ab28];
- 0.187 and 0.208 at $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792];
- 0.196, 0.193 and 0.167 at 1500 steps, data seed 1 [f-20261013-025812-b9ee].

**Force forms well after `accel_step`.**

- At $d_{\text{model}} = 128$, force formed persistently at steps 400 and 350, against step 100 for `accel_step`. From step 500 to 5000, force gained +0.074 and +0.081 in margin, while `accel_step` gained +0.025 and +0.022 [f-20261013-020028-8792].
- At $d_{\text{model}} = 64$ on data seed 1, force first held the bar at steps 400, 450 and 600. `accel_step` gained only 0.006–0.018 after step 600 [f-20261013-025812-b9ee]. That finding concludes the short-run "no force code" null is partly an artefact of where training stopped or of the learning-rate schedule horizon.
- At $e \le 0.8$ and $e \le 0.75$, force formed at steps 250–500 and was still rising at step 600 [f-20261013-000359-37e2, f-20261013-005923-526c].

![Formation timeline, d_model = 128, data seed 1, seed 80: force forms about 250 steps after accel_step](figure:formation/run-20261013-013406-a8e7)

![Force probe heatmap, N = 64, 12000 steps, seed 20](figure:heatmap/run-20261013-043205-0031/force)

**Which code leads depends on the metric.** Force often has the larger raw margin, because the `accel_step` control is high. Measured as a fraction of the available headroom, $(R^2 - c)/(1 - c)$, `accel_step` leads in every multi-seed comparison except the 2-layer model:

| Setting | `accel_step` | force |
|---|---|---|
| Wide $m_2$ range [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6-layer model [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41 |
| $N = 64$, 3000 steps [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $N = 64$, data seed 1 [f-20261013-054512-1b08] | 0.725, 0.740 | 0.44, 0.45 |
| $N = 128$, data seed 0 [f-20261013-051807-f3f1] | 0.67, 0.63 | ≈0.20, 0.18 |
| $N = 128$, data seed 1 [f-20261013-054542-23e1] | 0.70, 0.72 | 0.29, 0.30 |
| $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |
| $d_{\text{model}} = 64$, 1500 steps, data seed 1 [f-20261013-025812-b9ee] | 0.66–0.71 | 0.35–0.43 |

In the 2-layer model the comparison splits, one seed each way [f-20261013-000220-ab2d].

**Confound: force may partly be a read of distance.** The layer where force decodes best is unstable:

- it ranges from layer 0 to layer 4 [f-20261012-002713-8d93];
- it drifts across layers 1–6 in the 6-layer model [f-20261012-030233-ab28];
- it moved 4 → 1 → 4 during training in one seed [f-20261013-020028-8792];
- it differs between paired seeds (layer 1 vs 3) [f-20261013-054542-23e1].

In the 1500-step runs, force cleared the bar only after its best layer moved from 4 to 1, the layer where `distance` decodes best. Layer-4 force never exceeded about 0.085 there [f-20261013-025812-b9ee]. In most long runs force also peaks at layer 1, the `distance` layer [f-20261013-000220-ab2d, f-20261013-015914-3f49, f-20261013-051738-2a37, f-20261013-051807-f3f1]. Force is also collinear with `distance` and `accel_step` through $1/r^2$. Until a residualised probe is run, the force margin should be read as force decodability, not as evidence of a separate $m_2$-weighted force representation.

**Fixing the mass ratio did not help.** With $m_2 = 0.5$, the force law is exactly $|F| = 0.5/r^2$. Even so, force $R^2$ stayed at 0.750 and 0.732, while `distance` decoded at 0.87–0.89 [f-20261011-223000-b5f9]. The budget hard stop cut both runs short. Seed 14 reached only 0.55 completion, below the 80% replication bar, so the result rests mainly on seed 15 (0.917 completion).

### Sampling density

**Force decodability falls as sampling gets denser; normalised `accel_step` does not.**

- **$N = 16$ vs $N = 64$.** At 3000 steps the force margins match: 0.236–0.275 at $N = 16$ [f-20261013-000329-a86b] against 0.247 and 0.234 at $N = 64$ [f-20261013-015914-3f49].
- **$N = 128$, data seed 0, fresh seeds 22–23.** Force margins were 0.108 and 0.094. That is 0.139–0.140 below the matched $N = 64$ seeds, and seed 23 never reached the bar. Across the four $N = 128$ seeds the range is 0.094–0.150 [f-20261013-051807-f3f1]. The drop comes from lower trained $R^2$ (0.56–0.57 vs 0.68–0.70), not from higher controls.
- **$N = 128$, data seed 1.** Force margins were 0.137 and 0.145, against 0.209 at $N = 64$. That is a drop of 0.072 and 0.064, smaller than the roughly 0.10–0.14 drop on data seed 0 [f-20261013-054542-23e1, f-20261013-054512-1b08]. The direction replicates; the size depends on the dataset.
- **12000 steps.** The $N = 64$ force margin (0.255, 0.266) stays 0.062 and 0.069 above $N = 128$ (0.193, 0.197). In seed 21 the gap shrank from 0.091 at step 6000, so longer training might narrow it further [f-20261013-051738-2a37].

![Force probe heatmap, N = 128, seed 23: force never beats its control](figure:heatmap/run-20261013-050519-2b49/force)

**The original $N = 256$ reference did not reproduce, but the reruns are deterministic.** An earlier finding had concluded that force drops sharply between $N = 64$ and $N = 256$ [f-20261013-015914-3f49]. That rested on original $N = 256$ runs with force margins of 0.033–0.040. An exact rerun of the $N = 256$, 3000-step config gave different results [f-20261013-050513-2c7a]:

| Seed | force margin | `accel_step` margin |
|---|---|---|
| 20 | 0.113 ($R^2$ 0.644 vs control 0.531) | 0.162 |
| 21 | 0.003 (does not beat control) | 0.198 |

The original runs had `accel_step` margins of 0.231 and 0.245.

- **Schedule horizon ruled out.** At step 3000 the rerun matches 6000-step and 12000-step runs to within about 0.025, so the original runs are the outlier [f-20261013-050513-2c7a].
- **Second rerun identical.** A second identical rerun matched the first to 3 decimals at all 7 checkpoints, for all four probes. Under the current pipeline, training and probing are therefore deterministic for a fixed seed [f-20261013-054603-d6cc].
- **What follows.** The 0.11 force gap between seeds 20 and 21 at $N = 256$ is real seed-to-seed variance. The up-to-0.025 differences between run lengths come from the learning-rate schedule horizon [f-20261013-054603-d6cc].
- **Open question.** Why the original runs differ is unresolved. A higher `accel_step` control in the reruns (0.71–0.76 against 0.65–0.68) is suggested, but only inferred from summary numbers [f-20261013-050513-2c7a].

Taken together, the $N = 128$ arms now place force between $N = 64$ and $N = 256$ on both data seeds. The "sharp step" reading is no longer supported, although $N = 256$ force varies a lot between seeds.

![Force probe heatmap, N = 256 exact rerun, seed 21: force never beats its control](figure:heatmap/run-20261013-042340-4a86/force)

**Normalised `accel_step` margin across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.70 at $N = 64$ [f-20261013-015914-3f49];
- 0.725–0.740 at $N = 64$, data seed 1 [f-20261013-054512-1b08];
- 0.76 and 0.65 at $N = 64$, 12000 steps [f-20261013-051738-2a37];
- 0.63–0.67 at $N = 128$, data seed 0 [f-20261013-051807-f3f1];
- 0.70–0.72 at $N = 128$, data seed 1 [f-20261013-054542-23e1].

The $N = 128$ data seed 0 values are slightly lower. More seeds would show whether that is a real trend.

### Eccentricity sets the control level

As the training eccentricity range widens, both the control $R^2$ and the trained $R^2$ for `accel_step` fall:

| $e$ range | control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

Sources: [f-20261012-044142-d490, f-20261012-055051-b178, f-20261013-005923-526c, f-20261013-000359-37e2, f-20261012-034018-5526].

The decline is uneven. About 0.05 of $R^2$ is lost from $e \le 0.7$ to $e \le 0.75$, then about 0.19 from $e \le 0.75$ to $e \le 0.85$ [f-20261013-005923-526c]. All of these arms share data seed 0. Force stays much weaker than `accel_step` at every eccentricity ceiling, with normalised force margins of 0.16–0.22 at $e \le 0.7$–$0.8$ [f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### Linear decodability stops tracking prediction quality

In the 6-layer model at $e \le 0.85$, after step 2000 [f-20261012-030233-ab28]:

- layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 and from 0.683 to 0.705;
- over the same steps, `val_mse_ratio` fell 6.8× and 5.5×.

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- On high-$e$ systems, error was 16–18× the in-distribution error (`val_mse_ratio` 0.270 and 0.249). This fails the pre-registered 10× criterion, although the model still beats the baseline.
- Transfer to low mass ratios held at about 3.6× the in-distribution error.

**Trained on $e \in [0, 0.8]$,** the high-$e$ gap closed to about 1.05× and 1.01× [f-20261011-211551-d718]:

- The budget hard stop cut both runs short at step 2500 (completion 0.867 and 0.9), while `val_mse_ratio` was still falling.
- Force probe $R^2$ on held-out high-$e$ systems stayed low: 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **With strong probes** ($R^2$ 0.831 and 0.881), the fits to the `accel_step` direction were $r^{-1}$, of the form $a^2/r - c\,a$ with $c \approx 0.55$ (fit errors 0.245 and 0.201). The force fits were $0.81/r$ and $0.80/r$ [f-20261011-231118-7a37].
- **At $e \le 0.9$, with weak probes** ($R^2$ 0.395 and 0.503), the fits were also $r^{-1}$ [f-20261011-223027-8fb3].
- **Open check:** the fitter has not been run on the true `accel_step` target, so a fitting artefact cannot be ruled out [f-20261011-231118-7a37].

### Null and negative results

Ten hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** Force kept the larger raw margin [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×.** The gap was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 and 0.732, and the budget stop cut both runs short [f-20261011-223000-b5f9].
4. **Wide-$e$ training would yield an $r^{-2}$ symbolic fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **At $N = 64$, `accel_step` would clear a 0.15 bar and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow pre-registered window.** Only 2 of 5 seeds did, against the 4 required. Finer checkpoint spacing may explain part of the shift [f-20261012-054929-0067].
8. **Coarser sampling ($N = 16$) would raise the normalised `accel_step` margin.** It was 0.69 and 0.68 [f-20261013-000329-a86b], level with $N = 64$ [f-20261013-015914-3f49].
9. **Force margin would decline gradually from $N = 16$ to $N = 256$.** It was refuted because $N = 64$ sits at the $N = 16$ level [f-20261013-015914-3f49]. That verdict relied on the non-reproducing $N = 256$ runs. The later $N = 128$ arms show an intermediate force level [f-20261013-051807-f3f1, f-20261013-054542-23e1].
10. **An exact rerun of the $N = 256$, 3000-step config would reproduce the original margins within 0.02.** It did not, and the schedule-horizon explanation was also refuted [f-20261013-050513-2c7a]. A second rerun then reproduced the first exactly [f-20261013-054603-d6cc].

### Not yet tested, and limits

**Not yet reported:**

- `velocity` probe results;
- activation patching, or any other causal test of whether the `accel_step` or force codes are used;
- any force probe with `distance` and `accel_step` regressed out of the target;
- a check of the original $N = 256$ runs' per-layer $R^2$ to explain why they did not reproduce [f-20261013-050513-2c7a].

**Limits throughout:**

- All probes are linear.
- `formed_at_step` sometimes lags the trajectory by one checkpoint. Where the two disagree, the findings use the trajectory value [f-20261012-054929-0067, f-20261013-000329-a86b, f-20261013-054512-1b08].
- Four findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee, f-20261013-054512-1b08, f-20261013-054542-23e1]. All others use data seed 0.
- Determinism has been shown for one config on one machine [f-20261013-054603-d6cc].
- Most results come from a single 4-layer architecture.
