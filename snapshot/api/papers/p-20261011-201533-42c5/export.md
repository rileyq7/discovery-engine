# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated orbits (REBOUND, $G = M = 1$) encode the physical law $|F| = m_2/r^2$ or a task-specific substitute, and when any such structure forms. We fit linear probes to the residual stream for four targets: force, per-snapshot acceleration (`accel_step`, $(F/m_2)\,dt^2$), distance and energy. A probe counts only if it beats the best of several randomly initialised controls by a required margin. The lab has 41 replicated findings (28 supported, 13 refuted) from 99 evidence runs, 4 of which were cut short by the budget stop. `accel_step` beats its control in every reported run. At $N = 64$ it forms at steps 85–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its headroom-normalised margin stays at 0.63–0.79 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-234439-d851, f-20261013-051807-f3f1, f-20261013-063631-c4ed]. Force decodability falls with $N$. From $N = 64$ to $N = 128$, trained force $R^2$ drops 0.098–0.128 while the controls stay flat [f-20261013-075824-35f0], and the drop persists with $m_2$ fixed [f-20261013-234552-f8e8]. At $N = 256$, force fails its control in most seeds [f-20261013-063603-95b5, f-20261013-075749-3d2c]. Two negative results stand out: symbolic fits give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3], and transfer to high eccentricity failed, with error 16–18× in-distribution [f-20261011-203723-b7de].

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

The lab has 41 citable findings, all replicated. Of these, 28 are supported and 13 are refuted. Together they rest on 99 evidence runs, 4 of which were cut short by the budget hard stop. Unless stated otherwise, models have 4 layers and $d_{\text{model}} = 64$, are evaluated at $N = 64$ snapshots per orbit, and are probed with linear ridge probes. A probe result counts only if $R^2 - \text{control } R^2 \ge \max(0.1, \text{control spread})$. Findings from 11 October used a single random control. Later findings use best-of-3 or best-of-5 controls. In several places we report the headroom-normalised margin, $(R^2 - \text{control})/(1 - \text{control})$, because the `accel_step` control is often 0.7–0.8, which caps its raw margin.

### Prediction quality

Every model trained in-distribution beats constant-velocity extrapolation once it is trained:

- **Default config, $e \le 0.5$, 4000+ steps.** Final `val_mse_ratio` is about 0.011 [f-20261011-201533-0653, f-20261011-203628-5cd9].
- **3000-step $N = 64$ runs.** 0.014–0.018 [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08].
- **$N = 16$.** 0.0007 [f-20261013-000329-a86b].
- **6-layer, $d_{\text{model}} = 128$, $e \le 0.85$.** 0.0032–0.0037 [f-20261012-030233-ab28].
- **Short 300–600-step formation runs.** These end well above those levels: 0.41–0.53 at $e \le 0.5$ and 0.13–0.15 at $e \le 0.7$–$0.85$ [f-20261012-002713-8d93, f-20261012-054929-0067, f-20261012-034018-5526, f-20261012-055051-b178].
- **$N = 256$, $e \le 0.85$, 600 steps.** These runs never beat the baseline (2.03–2.37) [f-20261012-034134-5e82].
- **$N = 256$, $e \le 0.5$, 3000 steps.** These runs reach 0.24–0.39 [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c].

### When structure forms

**Distance and energy form first.** `distance` and `energy` beat their controls within the first 5–50 steps in every dense-checkpoint run, while `val_mse_ratio` is still far above 1 [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490]. This may partly be input geometry that is easy to read rather than learned dynamics [f-20261011-201533-0653].

**`accel_step` forms next, and before the model beats the baseline at $N = 64$.** This ordering holds in every $N = 64$, $e \le 0.5$ experiment:

| Checkpoint spacing | `accel_step` first ≥ 0.1 | `val_mse_ratio` first < 1 | Finding |
|---|---|---|---|
| 25 steps (seeds 4, 5) | step 100 (margins 0.115, 0.108) | step 175 | [f-20261011-203659-4e2f] |
| 10 steps (seeds 44–46) | steps 140, 140, 100 | steps 180, 180, 190 | [f-20261012-002713-8d93] |
| 5 steps, best-of-5 (seeds 47–49) | steps 85–90 | steps 135–165 | [f-20261012-005950-42a7] |
| 10 steps (seeds 62–64) | persistent at 90–100 | steps 140–160 | [f-20261012-044142-d490] |
| 5 steps (seeds 65–69) | persistent at 65–90 | — | [f-20261012-054929-0067] |

![Probe formation over training, N = 64, seed 64](figure:formation/run-20261012-043022-144a)

In the earliest runs, with 250-step checkpoints, `accel_step` and `distance` were already above their single controls at the first checkpoint (step 250). That is 1250 steps before `val_mse_ratio` first fell below 0.1 [f-20261011-201533-0653].

**The ordering depends on the baseline, not on when `accel_step` forms.** At $N = 16$, `val_mse_ratio` drops below 1 at step 20, while `accel_step` forms at steps 75–90, within 15 steps of the seed-matched $N = 64$ runs. The ordering flips because the constant-velocity baseline is weaker at $N = 16$ [f-20261012-015955-fc2b]. At $N = 256$ and $e \le 0.85$, `accel_step` forms at steps 60–70, matching $N = 64$ to within about 10 steps, even though those models never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526]. Formation therefore tracks a training-step window, not a fixed fraction of the loss drop or a fixed point relative to the baseline.

**Eccentricity range and learning rate shift formation only slightly.**

- **Eccentricity.** Formation sits at steps 50–100 at $e \le 0.7$, $0.75$, $0.8$ and $0.85$ [f-20261012-055051-b178, f-20261013-005923-526c, f-20261013-000359-37e2, f-20261012-034018-5526].
- **Learning rate.** Halving the learning rate moved formation from steps 60/60/70 to 70/80/100, a sub-proportional shift [f-20261012-044258-ab88].

**Force forms late, if at all.** Three regimes appear:

- **Brief early spike.** In every dense-checkpoint run, force shows a transient margin at step 10: 0.166–0.189 under best-of-5 controls at $N = 64$ [f-20261012-005950-42a7], 0.148–0.159 in seeds 62–64 [f-20261012-044142-d490], and 0.106–0.127 at $N = 16$ [f-20261012-015955-fc2b]. In each case it collapses to about 0 by steps 20–50.
- **No force code in short runs.** In 300-step runs at $N = 64$, $e \le 0.5$, force fails its control at the end in nearly every seed (final margins 0.000–0.080) [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490]. The exceptions are two borderline seeds at 0.103 and 0.105 [f-20261012-054929-0067].
- **Persistent force with longer training.** Force forms persistently at steps 400–600 on data seed 1 at 1500 steps [f-20261013-025812-b9ee], and at steps 350–400 with $d_{\text{model}} = 128$, against step 100 for `accel_step` [f-20261013-020028-8792]. The earlier "no force code" null was therefore partly an artefact of where training stopped [f-20261013-025812-b9ee].

### `accel_step` versus force at the end of training

`accel_step` beats its control in every evidence run, peaking at layer 3 or 4 (layer 6 in the 6-layer model). On raw margin, the two probes trade places depending on the setup:

- **Force leads on raw margin** in the default and wide-$m_2$ $e \le 0.5$ runs [f-20261011-201533-0653, f-20261011-203628-5cd9], at $N = 16$ [f-20261013-000329-a86b], at $N = 64$ after 3000 steps [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08], and in the 2-layer, $d_{\text{model}} = 32$ model [f-20261013-000220-ab2d].
- **`accel_step` leads on raw margin** at higher eccentricity [f-20261011-211551-d718, f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c], with $d_{\text{model}} = 128$ [f-20261012-030233-ab28, f-20261013-020028-8792], and at $N \ge 128$ [f-20261013-051807-f3f1, f-20261013-063603-95b5].

On headroom-normalised margin, `accel_step` leads in every 4- and 6-layer comparison:

| Setting | `accel_step` (normalised) | force (normalised) |
|---|---|---|
| Wide $m_2$, $e \le 0.5$ [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6 layers, $d_{\text{model}} = 128$, $e \le 0.85$ [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41, 0.41 |
| $N = 64$, seeds 22/23 [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $N = 64$, seeds 24–26 [f-20261013-075855-f75f] | 0.71, 0.69, 0.68 | ≈0.43–0.48 |
| $N = 64$, data seed 1 [f-20261013-054512-1b08] | 0.725, 0.740 | 0.44, 0.45 |
| $N = 64$, $m_2 = 0.5$ fixed (seeds 21, 20) [f-20261013-234439-d851] | 0.74, 0.79 | 0.48, 0.36 |
| 4 layers, $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |

The one exception is the 2-layer model, where each probe leads in one seed (seed 1: `accel_step` 0.46 vs force 0.36; seed 2: force 0.42 vs `accel_step` 0.37) [f-20261013-000220-ab2d].

**Force may be a read of distance.** At $N \le 128$, force usually peaks at layer 1, the same layer where `distance` is decoded best [f-20261013-015914-3f49, f-20261013-051738-2a37, f-20261013-075855-f75f, f-20261013-224908-8b46]. In the 1500-step data-seed-1 runs, layer-4 force never exceeded a margin of about 0.085 [f-20261013-025812-b9ee]. No probe with `distance` regressed out has been run, so we treat force margins as force *decodability*, not as evidence for a separate $m_2$-weighted representation.

**Linear decodability can stop tracking prediction quality.** In the 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$, `accel_step` $R^2$ levelled off at 0.70–0.74 while `val_mse_ratio` fell a further 5.5–6.8× [f-20261012-030233-ab28].

### The mass ratio

**Widening the mass ratio did not make force fall behind.** With $m_2 \in [0.01, 1]$, force kept the larger raw margin (0.243 and 0.235 vs `accel_step` 0.205 and 0.175) [f-20261011-203628-5cd9].

**Fixing $m_2$ did not lift force to 0.85.** In the first test, with $m_2 = 0.5$, force $R^2$ was 0.750 and 0.732 [f-20261011-223000-b5f9]. Both runs were cut short by the budget stop. Seed 14 completed 0.55 of its steps, below the 80% replication threshold, and seed 15 completed 0.917, so that result rests mainly on seed 15.

Complete 3000-step runs with $m_2 = 0.5$ at $N = 64$ now confirm the picture [f-20261013-234439-d851]:

- **Force still beats its control.** Seed 21: $R^2$ 0.789 vs control 0.594, margin 0.195, layer 1. Seed 20: 0.763 vs 0.630, margin 0.133, layer 4.
- **Higher trained $R^2$, smaller margins.** Trained force $R^2$ is higher than in the same seeds' wide-$m_2$ runs (0.726 / 0.737), but the margins are smaller (wide-$m_2$: 0.215 / 0.261). Fixing $m_2$ raises the force control from about 0.48–0.51 to 0.59–0.63.
- **No sign that $m_2$ adds decodable structure.** The smaller margin comes from random models finding the target easier to read, not from a missing $m_2$ factor.

![accel_step probe heatmap, N = 64, m₂ = 0.5, seed 21](figure:heatmap/run-20261013-231211-6833/accel_step)

### Sampling density: force decodability falls with $N$, `accel_step` does not

Final force margins after 3000 steps:

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | 0.215 / 0.261 [f-20261013-015914-3f49] | 0.150 / 0.131 [f-20261013-051807-f3f1] | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079 [f-20261013-075824-35f0] | 0.026 / 0.045 / 0.029 [f-20261013-075749-3d2c] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | ≈0.268 / ≈0.275 | 0.173 / 0.163 [f-20261013-224908-8b46] | — |
| $m_2 = 0.5$, seeds 20/21 | — | 0.133 / 0.195 [f-20261013-234439-d851] | 0.063 / 0.101 [f-20261013-234552-f8e8] | — |

The $N = 64$ values for data seed 1, seeds 22/23, come from the paired arm's rounded trajectory, as reported in [f-20261013-224908-8b46].

**$N = 16$ and $N = 64$ give the same force level.** A gradual decline across all $N$ was therefore refuted [f-20261013-000329-a86b, f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, the trained model loses force decodability while the controls barely move.**

- **Data seed 0, seeds 24–26.** Trained force $R^2$ fell by 0.098–0.128, controls moved by +0.002 to +0.009, and `distance` $R^2$ did not fall [f-20261013-075824-35f0].
- **Data seed 0, seeds 22/23.** The drop again sits in lower trained $R^2$ (0.56–0.57 vs 0.68–0.70), not in higher controls [f-20261013-051807-f3f1].
- **Data seed 1.** The margin drop was 0.064–0.072 for seeds 20/21 [f-20261013-054542-23e1]. For seeds 22/23, trained $R^2$ fell by 0.095–0.097 with controls changing by −0.002 to +0.017, while layer-1 `distance` held or rose [f-20261013-224908-8b46]. Within data seed 1, the spread between model seeds is as large as the difference between data seeds.
- **With $m_2$ fixed at 0.5.** The drop persists. Trained force $R^2$ fell from 0.763 / 0.789 to 0.693 / 0.684 (−0.070 / −0.105), controls changed by 0.012 or less, and seed 20 no longer beats its control. `accel_step` stayed strong (normalised 0.74 / 0.72), and layer-1 `distance` fell only slightly (0.950 / 0.926 → 0.915 / 0.908) [f-20261013-234552-f8e8]. The loss at $N = 128$ is therefore in reading out $1/r^2$, not in the $m_2$ factor. Seed 20's drop clears the pre-registered 0.06 bar by only 0.01, and the comparison is at matched step, not matched loss.
- **Long training does not close the gap.** After 12000 steps, $N = 64$ still sat 0.062 and 0.069 above $N = 128$ [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, the cause of the drop depends on the dataset.**

- **Data seed 1.** Trained force $R^2$ did not fall (about 0.66 → 0.671–0.672). Controls rose from 0.52 to 0.60–0.63, so the whole drop is a control effect [f-20261013-063631-c4ed].
- **Data seed 0, seeds 24–26.** Trained $R^2$ also fell, by 0.023–0.076, with controls accounting for roughly 35–50% of the margin loss [f-20261013-075749-3d2c].
- **Data seed 0, seeds 22/23.** For seed 23, roughly two thirds of the margin drop comes from the control [f-20261013-063603-95b5].

![Force probe heatmap, N = 256, seed 22](figure:heatmap/run-20261013-061904-9009/force)

**On data seed 0, the $N = 256$ shortfall is not just slower convergence.**

- **1000 more steps.** Training to 4000 steps raised trained force $R^2$ by only about 0.011–0.014. The gap to $N = 128$ stayed at 0.053–0.082, and force still failed its control [f-20261013-083108-2efd].
- **Matched on `val_mse_ratio`.** $N = 128$ still decoded force better: 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

Force was still gaining 0.005–0.007 margin per 500 steps, so a fixed ceiling has not been shown [f-20261013-083108-2efd].

**The normalised `accel_step` margin is flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.79 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08, f-20261013-234439-d851];
- 0.63–0.74 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-093210-2e03, f-20261013-224908-8b46, f-20261013-234552-f8e8];
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c, f-20261013-083108-2efd].

The per-snapshot acceleration code is equally readable at every sampling density tested. Linear force decodability falls with $N$ above 64 in every paired seed.

### Reproducibility

The original $N = 256$ reference runs did not reproduce. An exact rerun gave force margins of 0.113 and 0.003, against the original 0.033–0.040, and `accel_step` margins 0.033–0.083 away from the originals. At step 3000 the rerun agreed with the 6000- and 12000-step runs to within about 0.025, which rules out the learning-rate schedule horizon as the cause [f-20261013-050513-2c7a]. A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc].

Under the current pipeline, training and probing are therefore deterministic for a fixed seed on this machine. The 0.11 force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance. Why the original runs differed remains unexplained.

### Eccentricity sets the control level

Widening the training range of $e$ lowers both the `accel_step` control and the trained $R^2$ (all $N = 64$):

| $e$ range | control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

The decline steepens above $e \le 0.75$ [f-20261013-005923-526c]. Force is weaker at every ceiling, with normalised margins of 0.11–0.22, and forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$). Error was about 3.6× the in-distribution error, and force $R^2$ held at 0.75–0.77.
- **High eccentricity** ($e \in [0.5, 0.8]$). Error was 16–18× the in-distribution error, so the pre-registered 10× transfer criterion failed. The models still beat the baseline, and force $R^2$ fell to 0.40 / 0.34.
- **Three-body sets.** Ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$** [f-20261011-211551-d718]:

- **The high-$e$ gap closes.** High-$e$ error fell to 1.01–1.05× the in-distribution error.
- **Truncated runs.** Both runs were cut short by the budget stop at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- **The probes still differ.** With matched prediction error, force $R^2$ on held-out high-$e$ systems was 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **Strong probes** (`accel_step` $R^2$ 0.831 / 0.881). The probe direction fits as $a^2/r - c\,a$ with $c \approx 0.55$, not as the true $a^3/r^2$. Force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Weaker probes at $e \le 0.9$** ($R^2$ 0.40–0.50). These also fit as $r^{-1}$ [f-20261011-223027-8fb3].
- **Possible fitter artefact.** The fitter has not been run on the true `accel_step` target, so we cannot rule out that the fitter itself produces the $r^{-1}$ form.

### Null and negative results

Thirteen hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** It did not [f-20261011-203628-5cd9].
2. **Training on $e \le 0.5$ would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 / 0.732, in runs cut short by the budget stop [f-20261011-223000-b5f9].
4. **Wide-$e$ training would give an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar, and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$.** $N = 64$ matched $N = 16$ [f-20261013-015914-3f49].
10. **An exact $N = 256$ rerun would reproduce the original margins.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128 \to 256$ force drop comes from the controls alone.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 extra steps at $N = 256$ would close the gap to $N = 128$.** They did not [f-20261013-083108-2efd].
13. **Fixing $m_2$ would remove the $N = 64 \to 128$ force drop.** The drop persisted, at 0.070 / 0.105 [f-20261013-234552-f8e8].

Several supported findings are themselves nulls for force:

- Force fails its control at the end of short $N = 64$ runs [f-20261012-002713-8d93, f-20261012-044142-d490].
- Force fails its control at $N = 256$ in most seeds [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed].

### Not yet tested, and limits

The lab has not yet reported:

- `velocity` probe results;
- activation patching, or any other causal test of whether the decoded codes are used;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target;
- a matched-loss $N = 128$ vs $N = 256$ comparison on data seed 1;
- a matched-loss comparison for the $m_2$-fixed arms;
- an explanation of the original $N = 256$ discrepancy.

Several limits apply throughout:

- **Probes.** All probes are linear.
- **Formation steps.** `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-054512-1b08, f-20261013-000329-a86b].
- **Rounded values.** Several paired comparisons rely on 3-decimal trajectory values whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261013-083108-2efd, f-20261013-234552-f8e8].
- **Single controls in early runs.** The earliest findings used single random controls [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f]. Their `distance` and `energy` margins should be read against the up-to-0.2 spread that random models show on those targets.
- **Architecture.** Most results come from one architecture.
- **Datasets.** Six findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee, f-20261013-054512-1b08, f-20261013-054542-23e1, f-20261013-063631-c4ed, f-20261013-224908-8b46]. The $m_2$-fixed arms use two seeds [f-20261013-234439-d851, f-20261013-234552-f8e8].
