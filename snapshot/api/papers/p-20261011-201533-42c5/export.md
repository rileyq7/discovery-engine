# Discovery Engine: running draft

## Abstract

Do small transformers trained to predict simulated two-body orbits ($G = M = 1$) learn the gravitational law, or shortcuts that happen to work, and when does any structure form? We fit linear probes to the residual stream for physical force $|F| = m_2/r^2$, per-snapshot acceleration `accel_step` $= (2\pi/N)^2 a^3/((1+m_2)r^2)$, distance and energy, and judge each against several randomly initialised controls. Across 39 replicated findings (27 supported, 12 refuted; 95 evidence runs, 4 cut short by the budget stop), `accel_step` beats its control in every setting tested. At $N = 64$ it forms at steps 85–140 in six seeds, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its headroom-normalised margin stays at 0.63–0.75 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-075855-f75f, f-20261013-051807-f3f1, f-20261013-063631-c4ed]. Force decodability falls with sampling density instead: margins are 0.216–0.255, 0.079–0.143 and 0.026–0.045 at $N = 64$, 128 and 256 [f-20261013-075855-f75f, f-20261013-075824-35f0, f-20261013-075749-3d2c]. The main negative result is that symbolic fits to the probe directions give $r^{-1}$, not $r^{-2}$, in all four seeds tested. The fitter itself has not yet been checked on the true target [f-20261011-231118-7a37, f-20261011-223027-8fb3].

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

The lab's records contain 39 citable findings, all with status `replicated`. Of these, 27 support their pre-registered hypothesis and 12 refute it. Together they draw on 95 evidence runs. The budget hard stop cut 4 of those runs short, and we flag those runs wherever their numbers appear. Unless we say otherwise, the model has 4 layers and $d_{\text{model}} = 64$, is trained on data seed 0, and is probed with linear ridge probes. A probe "beats control" only if $R^2 - \text{control } R^2 \ge \max(0.1, \text{control spread})$. Older findings from 2026-10-11 report a single random control. Their `distance` and `energy` margins should therefore be read with caution, because random models differ by up to about 0.2 on those targets.

### Prediction quality

Every model trained to completion at $N \le 128$ beats constant-velocity extrapolation:

- **Default setting.** Seeds 2 and 3 reached a final `val_mse_ratio` of about 0.011 [f-20261011-201533-0653].
- **$N = 16$.** The ratio was 0.0007 [f-20261013-000329-a86b].
- **6 layers, $d_{\text{model}} = 128$, $e \le 0.85$.** The ratio was 0.0032–0.0037 [f-20261012-030233-ab28].

The exception is $N = 256$ at $e \le 0.85$. After 600 steps these models never beat the baseline, ending at 2.03–2.37 [f-20261012-034134-5e82]. At $e \le 0.5$, $N = 256$ models cross below 1 only around step 2000 and end at 0.24–0.32 after 3000 steps [f-20261013-063603-95b5, f-20261013-075749-3d2c].

### When structure forms

**Geometry forms first, then `accel_step`, then (sometimes) force.** In every timing experiment, `distance` and `energy` beat their controls within the first few checkpoints, at steps 5–50. At that point `val_mse_ratio` is still far above 1 [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490]. `accel_step` follows at layer 4. Its formation step is stable across seeds and checkpoint spacings:

| Setting | `accel_step` formation step | `val_mse_ratio` < 1 first at |
|---|---|---|
| 25-step checkpoints, seeds 4–5 [f-20261011-203659-4e2f] | 100, 100 | 175 |
| 10-step, seeds 44–46 [f-20261012-002713-8d93] | 140, 140, 100 | 180, 180, 190 |
| 10-step, best-of-5 controls, seeds 47–49 [f-20261012-005950-42a7] | 85, 85, 90 | 135–165 |
| 10-step, seeds 62–64 [f-20261012-044142-d490] | 90, 100, 100 | 140–160 |
| 5-step, seeds 65–69 [f-20261012-054929-0067] | 65–90 | — |

At $N = 64$ and $e \le 0.5$, then, the per-snapshot acceleration code becomes decodable 40–90 steps before the model beats the baseline. The margins at the crossing are thin, though: 0.100–0.115 against a bar of 0.1 [f-20261011-203659-4e2f, f-20261012-002713-8d93].

![Probe formation over training, N = 64, seed 46, 10-step checkpoints](figure:formation/run-20261011-234250-a7ca)

**The "before the baseline" ordering is not a stable marker.** Three experiments changed one factor and moved the formation step very little:

- **Sampling density.** At $N = 16$, `val_mse_ratio` falls below 1 at step 20, but `accel_step` still forms at steps 75–90. That is within 5–15 steps of the seed-matched $N = 64$ runs. The ordering flips because the baseline is weaker at $N = 16$, not because formation moves [f-20261012-015955-fc2b].
- **$N = 256$.** At $e \le 0.85$, `accel_step` forms at steps 60–70, the same steps as at $N = 64$, even though these models never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].
- **Learning rate.** Halving lr delayed formation by only 10–30 steps (to steps 70/80/100), not the roughly 2× that proportional scaling predicts [f-20261012-044258-ab88].

Formation therefore tracks a window of training steps more closely than it tracks loss relative to the baseline. A narrow pre-registered window for $N = 64$ formation was refuted: only 2 of 5 seeds fell inside it, and the misses were 5–10 steps early [f-20261012-054929-0067].

**An early force transient.** Every seed with 5–10-step checkpoints shows the same brief force signal: a margin of about 0.11–0.19 at steps 10–25 that falls to about 0 by steps 40–50. The transient survives best-of-5 controls [f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-054929-0067]. We don't know what it reflects.

**In short runs force never forms; in longer runs it forms late.**

- **Short runs (300 steps).** Force fails its control in most seeds. Final margins are 0.000–0.080 [f-20261012-002713-8d93, f-20261012-044142-d490], with borderline exceptions of 0.103 and 0.105 [f-20261012-054929-0067].
- **Longer runs.** Force crosses the bar at steps 350–600, against step 100–150 for `accel_step` [f-20261013-020028-8792, f-20261013-025812-b9ee].
- **The short-run null is partly a stopping artefact.** In 1500-step runs the step-600 force margin is already 0.104–0.138, while 600-step runs with the same seeds were at 0.085–0.099 [f-20261013-025812-b9ee].
- **Force keeps rising.** From step 500 to 5000 force gains 0.074–0.081, while `accel_step` gains only 0.022–0.025 [f-20261013-020028-8792].

### `accel_step` versus force

**Which code "leads" depends on the metric.** On raw margin, force often wins:

- 0.243/0.235 vs 0.205/0.175 with $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9];
- 0.221/0.255 vs 0.146/0.128 in a 2-layer, $d_{\text{model}} = 32$ model [f-20261013-000220-ab2d];
- 0.236/0.275 vs 0.131/0.140 at $N = 16$ [f-20261013-000329-a86b].

This happens because the `accel_step` control is high, around 0.65–0.81 at $e \le 0.5$. Normalised by headroom, $(R^2 - c)/(1 - c)$, `accel_step` leads in every multi-seed comparison except the 2-layer model, where each code wins in one seed [f-20261013-000220-ab2d]:

| Setting | `accel_step` (normalised) | force (normalised) |
|---|---|---|
| Wide $m_2$ range [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6 layers, $d_{\text{model}} = 128$, $e \le 0.85$ [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41, 0.41 |
| $N = 64$, seeds 22/23 [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $N = 64$, seeds 24–26 [f-20261013-075855-f75f] | 0.71, 0.69, 0.68 | ≈0.43–0.48 |
| $N = 64$, data seed 1 [f-20261013-054512-1b08] | 0.725, 0.740 | 0.44, 0.45 |
| $N = 64$, data seed 1, 1500 steps [f-20261013-025812-b9ee] | 0.66–0.71 | 0.35–0.43 |
| $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |

![accel_step probe heatmap, N = 64, seed 24](figure:heatmap/run-20261013-074422-d66a/accel_step)

**Confound: force may be a read of distance.** At $N \le 128$, force usually peaks at layer 1, where `distance` is also decoded best [f-20261013-015914-3f49, f-20261013-051738-2a37, f-20261013-075855-f75f, f-20261013-224908-8b46]. In the 1500-step data-seed-1 runs, layer-4 force never exceeded about 0.085 [f-20261013-025812-b9ee]. No probe with `distance` regressed out has been run. We therefore treat force margins as force *decodability*, not as evidence for a separate $m_2$-weighted representation.

**Fixing the mass ratio did not lift force.** With $m_2 = 0.5$, the force target is exactly $0.5/r^2$. Even so, force $R^2$ stayed at 0.750 and 0.732, while `distance` decoded at 0.87–0.89 [f-20261011-223000-b5f9]. Both runs were cut short by the budget stop. Seed 14 completed 0.55 of its steps, which is below the 80% replication threshold, and seed 15 completed 0.917. The result therefore rests mainly on seed 15.

**Width and depth.** In the 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$, linear `accel_step` $R^2$ levelled off at 0.70–0.74. Over the same steps `val_mse_ratio` fell a further 5.5–6.8×. Linear decodability thus stops tracking prediction quality [f-20261012-030233-ab28].

### Sampling density: force decodability falls with $N$, `accel_step` does not

Final force margins after 3000 steps:

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079 [f-20261013-075824-35f0] | 0.026 / 0.045 / 0.029 [f-20261013-075749-3d2c] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | ≈0.268 / ≈0.275 | 0.173 / 0.163 [f-20261013-224908-8b46] | — |

The $N = 64$ values for data seed 1, seeds 22/23, come from the paired arm's rounded trajectory as reported in [f-20261013-224908-8b46].

**$N = 16$ and $N = 64$ give the same force level**, so a gradual decline across all $N$ was refuted [f-20261013-000329-a86b, f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, the trained model loses force decodability.** The controls barely move:

- **Data seed 0, seeds 24–26.** Trained force $R^2$ fell by 0.098–0.128. Controls moved by +0.002 to +0.009, and `distance` $R^2$ did not fall [f-20261013-075824-35f0].
- **Data seed 0, seeds 22/23.** The same pattern holds [f-20261013-051807-f3f1].
- **Data seed 1, seeds 20/21.** The drop was smaller, 0.064–0.072 [f-20261013-054542-23e1].
- **Data seed 1, seeds 22/23 (new).** Trained force $R^2$ fell by 0.095–0.097, controls changed by −0.002 to +0.017, and layer-1 `distance` $R^2$ held or rose (0.933/0.935 vs 0.896/0.932). The loss is therefore specific to force [f-20261013-224908-8b46]. Within data seed 1, the spread between model seeds (0.064–0.072 vs 0.095–0.112 in margin) is as large as the difference between data seeds 0 and 1.
- **Long training does not close the gap.** After 12000 steps, $N = 64$ still sat 0.062 and 0.069 above $N = 128$ [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, the cause of the drop depends on the dataset.**

- **Data seed 1.** Trained force $R^2$ did not fall (about 0.66 → 0.671–0.672). Controls rose from 0.52 to 0.60–0.63, so the whole drop is a control effect [f-20261013-063631-c4ed].
- **Data seed 0, seeds 24–26.** Trained $R^2$ also fell, by 0.023–0.076, with controls accounting for roughly 35–50% of the margin loss [f-20261013-075749-3d2c].
- **Data seed 0, seeds 22/23.** The split is intermediate [f-20261013-063603-95b5].

![Force probe heatmap, N = 256, seed 22](figure:heatmap/run-20261013-061904-9009/force)

**On data seed 0, the $N = 256$ shortfall is not just slower convergence.**

- **1000 more steps.** Training to 4000 steps raised trained force $R^2$ by only about 0.011–0.014. The gap to $N = 128$ stayed at 0.053–0.082, and force still failed its control [f-20261013-083108-2efd].
- **Matched on `val_mse_ratio`.** Compared at a matched ratio, $N = 128$ still decoded force better: 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

This rests on two seeds on one dataset, and force was still gaining 0.005–0.007 margin per 500 steps. A fixed ceiling has not been shown [f-20261013-083108-2efd].

**The normalised `accel_step` margin is flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.74 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08];
- 0.63–0.72 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-093210-2e03, f-20261013-224908-8b46];
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c, f-20261013-083108-2efd].

The per-snapshot acceleration code is equally readable at every sampling density we tested. Linear force decodability falls with $N$ above 64 in every paired seed.

### Reproducibility

The original $N = 256$ reference runs did not reproduce. An exact rerun gave force margins of 0.113 and 0.003, against the original 0.033–0.040, and `accel_step` margins 0.033–0.083 away from the originals. At step 3000 the rerun agreed with the 6000- and 12000-step runs, which rules out the learning-rate schedule horizon as the cause [f-20261013-050513-2c7a]. A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. So under the current pipeline, training and probing are deterministic for a fixed seed. The 0.11 force gap between seeds 20 and 21 is genuine seed variance, and the original runs remain unexplained.

### Eccentricity sets the control level

Widening the training range of $e$ lowers both the `accel_step` control and the trained $R^2$ (all $N = 64$):

| $e$ range | control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

The decline steepens above $e \le 0.75$ [f-20261013-005923-526c]. Formation stays at steps 50–100 across these ranges [f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c, f-20261012-034018-5526]. Force is weaker at every ceiling, with normalised margins of 0.11–0.22, and it forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$). Error was about 3.6× the in-distribution error, and force $R^2$ held at 0.75–0.77.
- **High eccentricity** ($e \in [0.5, 0.8]$). Error was 16–18× the in-distribution error, so the pre-registered 10× transfer criterion failed. The models still beat the baseline, and force $R^2$ fell to 0.40/0.34.
- **Three-body sets.** Ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$** [f-20261011-211551-d718]:

- **High-$e$ error.** The gap closed to 1.01–1.05× the in-distribution error.
- **Truncated runs.** Both runs were cut short by the budget stop at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- **Probes still differ.** Even with matched prediction error, force $R^2$ on held-out high-$e$ systems was 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

The symbolic fits do not recover the inverse-square law:

- **Strong probes** (`accel_step` $R^2$ 0.831/0.881). The probe direction fits as $a^2/r - c\,a$ with $c \approx 0.55$, not as the true $a^3/r^2$. Force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Weaker probes at $e \le 0.9$** ($R^2$ 0.40–0.50). These also fit as $r^{-1}$ [f-20261011-223027-8fb3].
- **Possible fitter artefact.** The fitter has not been run on the true `accel_step` target, so we cannot rule out that it is the source of the $r^{-1}$ form.

### Null and negative results

Twelve hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** It did not [f-20261011-203628-5cd9].
2. **Training on $e \le 0.5$ would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750/0.732, in runs cut short by the budget stop [f-20261011-223000-b5f9].
4. **Wide-$e$ training would give an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$.** $N = 64$ matched $N = 16$ [f-20261013-015914-3f49].
10. **An exact $N = 256$ rerun would reproduce the original margins.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128 \to 256$ force drop comes from the controls alone.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 extra steps at $N = 256$ would close the gap to $N = 128$.** They did not [f-20261013-083108-2efd].

### Not yet tested, and limits

The lab has not yet reported:

- `velocity` probe results;
- activation patching, or any other causal test of whether the decoded codes are used;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target;
- a matched-loss $N = 128$ vs $N = 256$ comparison on data seed 1;
- an explanation of the original $N = 256$ discrepancy.

Several limits apply throughout:

- All probes are linear.
- `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-054512-1b08].
- Several paired comparisons rely on 3-decimal trajectory values, whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261013-083108-2efd].
- Most results come from one architecture.
- Six findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee, f-20261013-054512-1b08, f-20261013-054542-23e1, f-20261013-063631-c4ed, f-20261013-224908-8b46]. All others use data seed 0.
