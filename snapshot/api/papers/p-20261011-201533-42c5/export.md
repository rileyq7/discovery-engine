# Discovery Engine: running draft

## Abstract

Do small transformers that predict simulated orbits (REBOUND, $G = M = 1$) learn the physical law, or shortcuts that happen to work? And when does any internal structure form? We train next-step predictors and fit linear probes for force ($m_2/r^2$), per-snapshot acceleration (`accel_step`, $\propto a^3/r^2$), distance and energy. Each probe is judged against best-of-several random-init controls. This draft covers 25 replicated findings (16 supported, 9 refuted) from 64 runs; 4 runs were cut short by the budget hard stop. `accel_step` becomes decodable early, at steps 65–140, and at $N = 64$, $e \le 0.5$ this is often before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-054929-0067]. Its timing follows training steps rather than loss [f-20261012-015955-fc2b, f-20261012-034134-5e82]. Physical force does not beat its control in 300-step runs (margins 0.000–0.080) [f-20261012-002713-8d93, f-20261012-044142-d490]. With longer training it emerges late (margins 0.167–0.333) [f-20261013-025812-b9ee, f-20261012-030233-ab28], possibly as a read of distance. On headroom-normalised margin, `accel_step` still leads (e.g. 0.86 vs 0.49) [f-20261013-020028-8792]. The main negative result: in all four cited seeds, symbolic fits to the probe directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. The fitter has not yet been checked on the true target. Models trained on $e \le 0.5$ also transfer poorly to high-$e$ systems, with error 16–18× in-distribution [f-20261011-203723-b7de].

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

The lab has 25 citable findings so far, and all 25 are replicated across seeds. Of these, 16 support their hypothesis and 9 refute it. They rest on 64 evidence runs. The budget hard stop cut 4 of those runs short, and we flag them wherever their numbers appear.

### Prediction quality

The models beat constant-velocity extrapolation in almost every setting:

- **Default 4-layer, $d_{\text{model}} = 64$ model, about 4000–5000 steps:** final `val_mse_ratio` was about 0.011 [f-20261011-201533-0653].
- **$N = 16$, 3000 steps:** 0.0007 [f-20261013-000329-a86b].
- **6-layer, $d_{\text{model}} = 128$ model:** 0.0037 and 0.0032 [f-20261012-030233-ab28].
- **Short 300-step runs at $N = 64$:** these ended at 0.41–0.53 [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-054929-0067].
- **Exception, $N = 256$ with $e \in [0, 0.85]$:** the model never beat the baseline (final ratio 2.03–2.37) [f-20261012-034134-5e82].

### A per-snapshot acceleration code forms early and stays

A next-step predictor needs the per-snapshot acceleration $(F/m_2)\,dt^2 \propto a^3/r^2$, not the physical force $m_2/r^2$. The `accel_step` probe beat its random controls in every setting tested. With $e \le 0.5$, final layer-4 $R^2$ was 0.884–0.905 against best-of-$k$ controls of 0.740–0.786 [f-20261012-002713-8d93], and 0.890–0.902 against 0.724–0.746 [f-20261012-044142-d490].

**`accel_step` forms at a stable training step.** The margin first reached and held the 0.1 bar at:

- step 100, with 25-step checkpoints [f-20261011-203659-4e2f];
- steps 100–140 [f-20261012-002713-8d93];
- steps 85–90 [f-20261012-005950-42a7];
- steps 90–100 [f-20261012-044142-d490];
- steps 65–90, with 5-step checkpoints [f-20261012-054929-0067];
- step 100 in a $d_{\text{model}} = 128$ model on data seed 1 [f-20261013-020028-8792].

At $N = 64$ and $e \le 0.5$, these crossings came 40–90 steps before `val_mse_ratio` first fell below 1 [f-20261012-002713-8d93, f-20261012-044142-d490]. With sparse 250-step checkpoints, `accel_step` had already formed at the first trained checkpoint (step 250, `val_mse_ratio` 0.87). That is 1250 steps before the ratio fell below 0.1 [f-20261011-201533-0653].

**`distance` and `energy` become readable earlier still**, at steps 10–30, while the model is far worse than the baseline (`val_mse_ratio` 5 to 83) [f-20261011-203659-4e2f, f-20261012-002713-8d93]. Random models disagree substantially on these two targets, so their margins have a higher bar.

**Formation tracks training steps, not loss.**

- **$N = 16$:** formation came at steps 90, 75 and 80, within 5–15 steps of the seed-matched $N = 64$ runs. But `val_mse_ratio` fell below 1 at step 20, so the order "forms before beating the baseline" reverses [f-20261012-015955-fc2b].
- **$N = 256$:** formation came at steps 60, 70 and 60, against 60–70 for the paired $N = 64$ arm, even though this model never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].
- **Loss at formation varies:** 2.0–2.9% of the step-0 loss at $N = 64$, $e \le 0.5$ [f-20261012-044142-d490]; 8–10% at $e \le 0.85$ [f-20261012-034018-5526]; about 15–18% at $N = 256$ [f-20261012-034134-5e82].
- **Learning rate:** halving it delayed formation by only +10, +20 and +30 steps [f-20261012-044258-ab88].

![Formation timeline, N = 16, seed 47: val_mse_ratio drops below 1 before accel_step forms](figure:formation/run-20261012-015504-ae8c)

![accel_step probe heatmap, N = 64, seed 62](figure:heatmap/run-20261012-043022-3319/accel_step)

**Caveat on thin margins.** Where the `accel_step` control is high (0.71–0.79 at $e \le 0.5$), final margins sit close to the bar: 0.109–0.158 [f-20261012-002713-8d93]. In two of three seeds the margin never reached 0.15 [f-20261012-005950-42a7]. Formation steps therefore depend on the threshold.

### Physical force: absent in short runs, late and growing in long ones

**In 300-step runs, force mostly did not beat its control.** Final force margins were:

- 0.040, 0.000 and 0.062 [f-20261012-002713-8d93];
- 0.039–0.059 [f-20261012-005950-42a7];
- 0.033–0.080 [f-20261012-044142-d490];
- 0.065–0.078 at half the learning rate [f-20261012-044258-ab88];
- 0.004–0.031 at $N = 256$ [f-20261012-034134-5e82].

A few seeds were borderline exceptions: 0.103 and 0.105 in 2 of 5 seeds [f-20261012-054929-0067], and 0.123 in one $N = 16$ seed [f-20261012-015955-fc2b].

**Every short run also showed a brief early force spike.** It peaked at a margin of about 0.11–0.19 around steps 10–25 and fell to about 0 by steps 40–50 [f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-054929-0067]. Best-of-5 controls did not remove it [f-20261012-005950-42a7].

**With longer training, a force code appears.** After 3000–5000 steps, force margins were:

- 0.223 and 0.230 [f-20261011-201533-0653];
- 0.243 and 0.235 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9];
- 0.247 and 0.234 at $N = 64$ [f-20261013-015914-3f49];
- 0.236 and 0.275 at $N = 16$ [f-20261013-000329-a86b];
- 0.221 and 0.255 in a 2-layer, $d = 32$ model [f-20261013-000220-ab2d];
- 0.300 and 0.333 in the 6-layer model [f-20261012-030233-ab28];
- 0.187 and 0.208 in the $d_{\text{model}} = 128$ model on data seed 1 [f-20261013-020028-8792].

**Force forms well after `accel_step`.** In the $d_{\text{model}} = 128$ runs, force formed persistently at steps 400 and 350, against step 100 for `accel_step`. From step 500 to step 5000, force gained +0.074 and +0.081 in margin, while `accel_step` gained only +0.025 and +0.022 [f-20261013-020028-8792].

**The newest result** repeats this at $d_{\text{model}} = 64$ on data seed 1 [f-20261013-025812-b9ee]:

- Training to 1500 steps made force beat its control persistently in all 3 seeds. It first held the bar at steps 400, 450 and 600, and final margins were 0.196, 0.193 and 0.167.
- Over the same period `accel_step` gained only 0.006–0.018 after step 600.
- The finding reports that at step 600 these runs already had force margins of 0.104–0.138. Earlier 600-step runs with the same seeds and data were at 0.085–0.099 at the same step. So the short-run "no force code" null is partly an artefact of where training stopped or of the learning-rate schedule horizon.

In other settings force also formed late, at steps 250–500 in 600-step runs at $e \le 0.8$ and $e \le 0.75$, and was still rising when training ended [f-20261013-000359-37e2, f-20261013-005923-526c].

![Formation timeline, d_model = 128, data seed 1, seed 80: force forms about 250 steps after accel_step](figure:formation/run-20261013-013406-a8e7)

![Force probe heatmap, d_model = 64, data seed 1, seed 75, 1500 steps](figure:heatmap/run-20261013-025012-5dee/force)

**Which code leads depends on the metric.** Force often has the larger raw margin. Measured as a fraction of available headroom, $(R^2 - c)/(1 - c)$, `accel_step` leads in almost every case:

| Setting | `accel_step` | force |
|---|---|---|
| Wide $m_2$ range [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6-layer model [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41 |
| $N = 64$, 3000 steps [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |
| $d_{\text{model}} = 64$, data seed 1 [f-20261013-025812-b9ee] | 0.66–0.71 | 0.35–0.43 |

The 2-layer model is split, with one seed each way [f-20261013-000220-ab2d].

**Confound: force may partly be a read of distance.** The layer where force is best decoded is unstable. It ranges from layer 0 to layer 4 [f-20261012-002713-8d93], drifts across layers 1–6 in the 6-layer model [f-20261012-030233-ab28], and moved 4 → 1 → 4 during training in one seed [f-20261013-020028-8792]. In the newest runs, force cleared the bar only after its best layer moved from 4 to 1, which is the layer where `distance` is decoded best. Layer-4 force never exceeded about 0.085 there [f-20261013-025812-b9ee]. Force also peaks at the `distance` layer in other runs [f-20261013-000220-ab2d, f-20261013-015914-3f49], and it is collinear with `distance` and `accel_step` through $1/r^2$. Part of the force margin may therefore read distance rather than an $m_2$-weighted force.

**Fixing the mass ratio did not help.** With $m_2$ fixed at 0.5, force $R^2$ stayed at 0.750 and 0.732, while `distance` was decoded at 0.87–0.89 [f-20261011-223000-b5f9]. The budget hard stop cut both of these runs short. Seed 14 reached only 0.55 completion, below the 80% replication bar, so the result rests mainly on seed 15 (0.917).

### Sampling density

**Force drops off sharply between $N = 64$ and $N = 256$, not gradually.** At 3000 steps, $N = 64$ force margins (0.247 and 0.234) match the $N = 16$ level (0.236–0.275) and are far above $N = 256$ (0.033–0.040) [f-20261013-015914-3f49].

**The normalised `accel_step` margin is flat across $N$:** 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b] and 0.68–0.70 at $N = 64$ [f-20261013-015914-3f49].

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

The decline is uneven. About 0.05 of $R^2$ is lost from $e \le 0.7$ to $e \le 0.75$, then about 0.19 from $e \le 0.75$ to $e \le 0.85$ [f-20261013-005923-526c]. All these arms share data seed 0.

### Linear decodability stops tracking prediction quality

In the 6-layer model at $e \le 0.85$, after step 2000 [f-20261012-030233-ab28]:

- Layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 and from 0.683 to 0.705.
- Over the same steps, `val_mse_ratio` fell 6.8× and 5.5×.

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- Error on high-$e$ systems was 16–18× the in-distribution error (`val_mse_ratio` 0.270 and 0.249). This fails the pre-registered 10× criterion, although the model still beats the baseline.
- Transfer to low mass ratios held at about 3.6× in-distribution.

**Trained on $e \in [0, 0.8]$**, the gap closed to about 1.05× and 1.01× [f-20261011-211551-d718]:

- The budget hard stop cut both runs short at step 2500 (completion 0.867 and 0.9), while `val_mse_ratio` was still falling.
- Force probe $R^2$ on held-out high-$e$ systems stayed low: 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **With strong probes** ($R^2$ 0.831 and 0.881), the fits to the `accel_step` direction were $r^{-1}$, of the form $a^2/r - c\,a$ with $c \approx 0.55$ (fit errors 0.245 and 0.201). The force fits were $0.81/r$ and $0.80/r$ [f-20261011-231118-7a37].
- **At $e \le 0.9$, with weak probes** ($R^2$ 0.395 and 0.503), the fits were also $r^{-1}$ [f-20261011-223027-8fb3].
- **Open check:** the fitter has not yet been run on the true `accel_step` target, so a fitting artefact cannot be ruled out [f-20261011-231118-7a37].

### Null and negative results

Nine hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** Force kept the larger raw margin [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×.** The gap was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 and 0.732, and both runs were cut short by the budget stop [f-20261011-223000-b5f9].
4. **Wide-$e$ training would yield an $r^{-2}$ symbolic fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **At $N = 64$, `accel_step` would clear a 0.15 bar and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow pre-registered window.** Only 2 of 5 seeds did, and at least 4 were required. Finer checkpoint spacing may explain part of the shift [f-20261012-054929-0067].
8. **Coarser sampling ($N = 16$) would raise the normalised `accel_step` margin.** It was 0.69 and 0.68 [f-20261013-000329-a86b], level with $N = 64$ [f-20261013-015914-3f49].
9. **Force margin would decline gradually with $N$.** $N = 64$ instead sits at the $N = 16$ level [f-20261013-015914-3f49].

### Not yet tested, and limits

**Not yet reported:**

- `velocity` probe results;
- activation patching, or any other causal test of whether the `accel_step` or force codes are used;
- any force probe with `distance` and `accel_step` regressed out of the target.

**Limits throughout:**

- All probes are linear.
- `formed_at_step` sometimes lags the trajectory by one checkpoint. Where the two disagree, the findings use the trajectory value [f-20261012-054929-0067, f-20261013-000329-a86b].
- Two findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee]. All others use data seed 0.
- Most results come from a single 4-layer architecture.
