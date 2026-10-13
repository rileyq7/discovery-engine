# Discovery Engine: running draft

## Abstract

When a small transformer predicts two-body orbits well, does its residual stream contain Newtonian gravity, $|F| = m_2/r^2$, or a substitute tied to how the orbits are sampled? We train transformers on REBOUND orbits sampled at $N$ snapshots per orbit. We then fit linear probes for force and for the per-snapshot acceleration $(F/m_2)\,dt^2$ (`accel_step`), and judge each probe against several randomly initialised controls. The lab has 37 replicated findings: 25 supported and 12 refuted, from 91 runs, of which 4 were cut short by the budget stop. `accel_step` beats its control in every setting. At $N = 64$ it forms before the model beats constant-velocity extrapolation [f-20261012-002713-8d93]. Its headroom-normalised margin stays at 0.63–0.75 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-054512-1b08, f-20261013-051807-f3f1, f-20261013-075749-3d2c]. Force decodability falls with $N$. For the same seeds, final margins are 0.216–0.255 at $N = 64$, 0.079–0.143 at $N = 128$ and 0.026–0.045 at $N = 256$ [f-20261013-075855-f75f, f-20261013-075824-35f0, f-20261013-075749-3d2c]. The main negative results are twofold. Symbolic fits to probe directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. And 1000 extra steps did not close the force gap between $N = 256$ and $N = 128$ [f-20261013-083108-2efd]. All probes are linear, and no causal test has been run.

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

The lab has 37 citable findings, and all of them are replicated across seeds. Of these, 25 support their hypothesis and 12 refute it. They rest on 91 evidence runs. The budget hard stop cut 4 of those runs short, and each is flagged where its numbers are used.

The main result so far is that the models encode a per-snapshot acceleration. In almost every configuration, a linear probe reads `accel_step` $= (2\pi/N)^2 a^3 / ((1+m_2) r^2)$ above its random controls. That code forms early in training and stays equally readable across sampling densities. Physical force $|F| = m_2/r^2$ is different. It forms late, it is weaker relative to its headroom, its decodability falls as `steps_per_orbit` ($N$) grows, and it may partly be a read of distance. Symbolic fits to the probe directions have not recovered the $r^{-2}$ law.

### Prediction quality

Trained models beat constant-velocity extrapolation by wide margins:

- **Default 4-layer, $d_{\text{model}} = 64$, $N = 64$, $e \le 0.5$.** Final `val_mse_ratio` is about 0.011 [f-20261011-201533-0653], and 0.014–0.016 on fresh seeds [f-20261013-075855-f75f].
- **$N = 16$, 3000 steps.** `val_mse_ratio` is 0.0007 [f-20261013-000329-a86b].
- **Larger models.** A 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$ reaches 0.0032–0.0037 [f-20261012-030233-ab28]. A 4-layer, $d_{\text{model}} = 128$ model reaches 0.005–0.006 after 5000 steps [f-20261013-020028-8792].

At fine sampling the baseline is much harder to beat:

- At $N = 256$ and $e \le 0.85$, 600-step models never beat it (final `val_mse_ratio` 2.03–2.37) [f-20261012-034134-5e82].
- At $N = 256$, 3000-step models end at 0.24–0.32 [f-20261013-063603-95b5, f-20261013-075749-3d2c].

### Per-snapshot acceleration forms early

In short runs with dense checkpoints, the `accel_step` margin reaches the required 0.1 bar early, mostly before the model beats the baseline:

- **$N = 64$, 25-step checkpoints.** `accel_step` formed at step 100 in both seeds, while `val_mse_ratio` was still 1.96 and 1.84. It first fell below 1 at step 175 [f-20261011-203659-4e2f].
- **$N = 64$, 10-step checkpoints.** Formation came at steps 140, 140 and 100, which is 40–90 steps before `val_mse_ratio` < 1 (steps 180–190) [f-20261012-002713-8d93].
- **Best-of-5 controls.** Formation came at steps 85–90, 45–75 steps before `val_mse_ratio` < 1 [f-20261012-005950-42a7]. A further three seeds formed at steps 90–100 [f-20261012-044142-d490].
- **Longer run, sparse checkpoints.** In the default config, gravity-related probes already beat their controls at the first checkpoint (step 250). That is 1250 steps before `val_mse_ratio` first fell below 0.1 [f-20261011-201533-0653].

![Probe formation over training, N = 64, seed 46](figure:formation/run-20261011-234250-a7ca)

The formation step is largely a property of training step. It does not track loss level:

- **$N = 16$.** `val_mse_ratio` drops below 1 at step 20, but `accel_step` forms at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs. The "forms before beating baseline" ordering flips only because the baseline is weaker at $N = 16$, not because formation moves [f-20261012-015955-fc2b].
- **$N = 256$, $e \le 0.85$.** `accel_step` forms at steps 60–70, the same window as $N = 64$, even though these models never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].
- **Halving the learning rate.** Formation was delayed by only 10–30 steps [f-20261012-044258-ab88].

Two caveats apply to the timing:

- **Thin margins.** At $e \le 0.5$ the `accel_step` controls are high (about 0.71–0.79), so margins sit only just above 0.1. A stricter bar would move or remove several crossings [f-20261012-002713-8d93, f-20261012-005950-42a7].
- **Window too narrow.** A pre-registered $N = 64$ formation window was met by only 2 of 5 seeds at 5-step resolution. The misses were one checkpoint early [f-20261012-054929-0067].

### Physical force: a transient, then a late and weaker code

**Early transient.** Force shows a short-lived spike early in training. Its margin is 0.166–0.189 at step 10 and decays to about 0 by steps 40–50; this survives best-of-5 controls [f-20261012-005950-42a7]. The same spike appears in the other short-run experiments [f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-054929-0067].

**Short runs.** In 300-step runs at $N = 64$, force does not beat its control at the end [f-20261012-002713-8d93, f-20261012-044142-d490]. In 2 of 5 seeds it scraped past the bar by margins of 0.103 and 0.105 [f-20261012-054929-0067].

**Longer runs.** With more training, a force code forms late and keeps strengthening:

- **1500 steps.** Force formed at steps 400–600. The earlier "no force code" null was therefore partly an artefact of where training stopped [f-20261013-025812-b9ee].
- **$d_{\text{model}} = 128$.** Force formed at steps 400 and 350, against step 100 for `accel_step` [f-20261013-020028-8792].
- **3000 steps at $N = 64$.** Final margins were 0.247 and 0.234 [f-20261013-015914-3f49]. On fresh seeds 24–26 they were 0.228, 0.255 and 0.216 [f-20261013-075855-f75f].
- **12000 steps.** Final margins were 0.255 and 0.266 [f-20261013-051738-2a37].

![Force probe heatmap, N = 64, seed 24](figure:heatmap/run-20261013-074422-d66a/force)

**Which code leads depends on the metric.** Force often has the larger raw margin because the `accel_step` control is high. On headroom-normalised margin, $(R^2 - R^2_{\text{control}})/(1 - R^2_{\text{control}})$, `accel_step` leads in every multi-seed comparison below:

| Setting | `accel_step` | force |
|---|---|---|
| Wide $m_2$ range [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6-layer model [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41 |
| $N = 64$, seeds 22/23 [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $N = 64$, seeds 24–26 [f-20261013-075855-f75f] | 0.71, 0.69, 0.68 | ≈0.43–0.48 |
| $N = 64$, data seed 1 [f-20261013-054512-1b08] | 0.725, 0.740 | 0.44, 0.45 |
| $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |

The 2-layer, $d_{\text{model}} = 32$ model is the exception: on normalised margin, one seed favours each code [f-20261013-000220-ab2d].

**Confound: force may partly be a read of distance.** At $N \le 128$, force usually peaks at layer 1, the `distance` layer [f-20261013-015914-3f49, f-20261013-051738-2a37, f-20261013-075855-f75f]. In the 1500-step runs, layer-4 force never exceeded about 0.085 [f-20261013-025812-b9ee]. No residualised probe has been run yet, so the force margin should be read as force *decodability*, not as a separate $m_2$-weighted representation.

**Fixing the mass ratio did not lift force.** With $m_2 = 0.5$, so that $|F| = 0.5/r^2$ exactly, force $R^2$ stayed at 0.750 and 0.732, while `distance` decoded at 0.87–0.89 [f-20261011-223000-b5f9]. The budget stop cut both runs short. Seed 14 reached only 0.55 completion, below the 80% replication bar, so the result rests mainly on seed 15 (0.917 completion).

### Sampling density: force decodability falls with $N$, `accel_step` does not

Final force margins at 3000 steps:

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079 [f-20261013-075824-35f0] | 0.026 / 0.045 / 0.029 [f-20261013-075749-3d2c] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |

**$N = 16$ and $N = 64$ give the same force level** [f-20261013-000329-a86b, f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, the drop comes from the trained model.**

- **Seeds 24–26.** Trained force $R^2$ fell by 0.098, 0.106 and 0.128, while the controls moved by only +0.002 to +0.009. `distance` $R^2$ did not fall [f-20261013-075824-35f0].
- **Seeds 22/23.** The same pattern holds [f-20261013-051807-f3f1].
- **Data seed 1.** The same pattern holds, but the drop is smaller (0.064–0.072) [f-20261013-054542-23e1].
- **12000 steps.** The $N = 64$ margin still sat 0.062 and 0.069 above $N = 128$ [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, the cause of the drop depends on the dataset.**

- **Data seed 1.** Trained force $R^2$ did not fall (about 0.66 at $N = 128$, 0.671–0.672 at $N = 256$). The controls rose from 0.52 to 0.60–0.63, so the whole drop is a control effect [f-20261013-063631-c4ed].
- **Data seed 0, seeds 24–26.** Trained force $R^2$ fell by 0.054–0.076, 0.054–0.062 and 0.023–0.028. The controls account for roughly 35–50% of the margin loss. This refuted the hypothesis that the drop is driven by the controls alone [f-20261013-075749-3d2c].
- **Data seed 0, seeds 22/23.** The split is intermediate: trained $R^2$ fell 0.044 and 0.021 [f-20261013-063603-95b5].
- **Caveats.** The paired $N = 128$ values for seeds 24–26 come from 3-decimal trajectories, whose $R^2$ field may be a maximum over layers. Force was also still rising at step 3000 [f-20261013-075749-3d2c, f-20261013-075855-f75f].

**1000 more training steps at $N = 256$ did not close the gap.** Seeds 24 and 25 were trained to 4000 steps. Final trained force $R^2$ was 0.557 and 0.569, still 0.082 and 0.053 below the same seeds' $N = 128$ values (0.639, 0.621). Between steps 3000 and 4000, trained force $R^2$ rose by only about 0.011 and 0.014, and force still failed its control in both seeds. `accel_step` held $R^2$ 0.906–0.915 against controls of 0.684–0.716, a margin of 0.199–0.223 and a normalised margin of about 0.70 [f-20261013-083108-2efd].

This rules out a gap that closes within 1000 extra steps. It does not show a fixed ceiling: force was still gaining about 0.005–0.007 margin per 500 steps, and `val_mse_ratio` was still falling. Seed 25's gain also sits just under the pre-registered 0.015 bar.

![accel_step probe heatmap, N = 256, seed 24, 4000 steps](figure:heatmap/run-20261013-081947-a102/accel_step)

**The normalised `accel_step` margin is flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.71 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f], and 0.725–0.740 on data seed 1 [f-20261013-054512-1b08];
- 0.63–0.70 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0], and 0.70–0.72 on data seed 1 [f-20261013-054542-23e1];
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c, f-20261013-083108-2efd].

So the per-snapshot acceleration code stays equally readable from $N = 16$ to $N = 256$. Linear force decodability, by contrast, falls with $N$ in every paired seed.

**The original $N = 256$ reference did not reproduce, but reruns are deterministic.** An exact rerun of the original config gave force margins of 0.113 and 0.003, against 0.033–0.040 originally. At step 3000 it matched the 6000- and 12000-step runs, which rules out the learning-rate schedule horizon as the cause [f-20261013-050513-2c7a]. A second rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. Why the original runs differ is still unexplained.

### Eccentricity sets the control level

Widening the training eccentricity range lowers both the control and the trained `accel_step` $R^2$:

| $e$ range | control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

Sources: [f-20261012-044142-d490, f-20261012-055051-b178, f-20261013-005923-526c, f-20261013-000359-37e2, f-20261012-034018-5526].

The decline steepens above $e \le 0.75$ [f-20261013-005923-526c]. Force stays much weaker at every ceiling, with normalised margins of 0.16–0.22 [f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

In a 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$, linear `accel_step` $R^2$ levelled off at about 0.70–0.74. Over the same training, `val_mse_ratio` fell a further 5.5–6.8×. Linear decodability therefore stops tracking prediction quality [f-20261012-030233-ab28].

### Out-of-distribution transfer

**Trained on $e \le 0.5$:**

- Error on high-$e$ systems was 16–18× the in-distribution error. This fails the pre-registered 10× criterion, although the model still beats the baseline.
- Transfer to low mass ratios held at about 3.6× [f-20261011-203723-b7de].

**Trained on $e \in [0, 0.8]$:**

- The high-$e$ gap closed to about 1.0×.
- The budget stop cut both runs short at step 2500 (completion 0.867 and 0.9).
- Force probe $R^2$ on held-out high-$e$ systems stayed at 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls [f-20261011-211551-d718].

### Symbolic regression has not recovered $r^{-2}$

- **Strong probes** ($R^2$ 0.831 and 0.881). The `accel_step` direction fits as $a^2/r - c\,a$ with $c \approx 0.55$, and force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Weak probes at $e \le 0.9$.** The fits were also $r^{-1}$ [f-20261011-223027-8fb3].
- **Open check.** The fitter has not been run on the true `accel_step` target, so a fitting artefact cannot be ruled out.

### Null and negative results

Twelve hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** Force kept the larger raw margin [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×.** The gap was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 and 0.732, and both runs were cut short [f-20261011-223000-b5f9].
4. **Wide-$e$ training would give an $r^{-2}$ symbolic fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow pre-registered window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It stayed level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$ to $N = 256$.** This was refuted because $N = 64$ matched $N = 16$ [f-20261013-015914-3f49]. Later paired arms show a steady decline from $N = 64$ upward [f-20261013-075824-35f0, f-20261013-075749-3d2c].
10. **An exact $N = 256$ rerun would reproduce the original margins within 0.02.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128$ → $N = 256$ force drop is driven by the controls alone.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 more training steps at $N = 256$ would close the trained force $R^2$ gap to $N = 128$.** The gap stayed at 0.053–0.082 [f-20261013-083108-2efd].

### Not yet tested, and limits

**Not yet reported:**

- `velocity` probe results;
- activation patching, or any other causal test of whether the codes are used;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- symbolic regression on the true `accel_step` target;
- an explanation of why the original $N = 256$ runs differ.

**Limits throughout:**

- All probes are linear.
- `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-054512-1b08].
- Most results come from one 4-layer, $d_{\text{model}} = 64$ architecture.
- Five findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee, f-20261013-054512-1b08, f-20261013-054542-23e1, f-20261013-063631-c4ed]. All others use data seed 0.
