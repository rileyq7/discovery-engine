# Discovery Engine: running draft

## Abstract

Do small transformers trained to predict two-body orbits (REBOUND, $G = M = 1$) learn the gravitational law, or shortcuts that happen to work? We train them on next-step prediction and fit linear probes to their residual streams. The probe targets are physical force ($m_2/r^2$), per-snapshot acceleration (`accel_step`, $\propto a^3/r^2$), distance and energy, and each probe is judged against best-of-several random-model controls. Across 52 replicated findings (39 supported, 13 refuted; 123 evidence runs, 4 cut short by the budget stop), `accel_step` forms early, at steps 100–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93]. Physical force is weaker and depends on sampling. Its margin is 0.215–0.261 at $N = 64$ snapshots per orbit but only 0.033–0.040 at $N = 256$ [f-20261013-015914-3f49, f-20261013-063603-95b5]. From $N = 64$ to $128$, trained force $R^2$ falls by 0.098–0.128 while controls stay flat [f-20261013-075824-35f0]. With $m_2$ fixed, the gap persists at 12000 steps [f-20261014-051311-3e2e]. Wider eccentricity caps linear decodability: at $e \le 0.9$, `accel_step` $R^2$ settles at 0.421–0.457 [f-20261016-033018-5b87]. The main negative result is that symbolic regression on probe directions gives $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. The fitter has not yet been checked on the true target.

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

The lab has 52 citable findings. All 52 are replicated: each rests on at least two completed seeds that agree. Of these, 39 support their hypothesis and 13 refute it. Together they draw on 123 evidence runs. The budget hard stop cut 4 of those runs short, and they are flagged where their numbers appear.

Unless noted, models have 4 layers and $d_{\text{model}} = 64$. Every probe is judged against random-model controls, and a probe counts only if $R^2 - \text{control } R^2 \ge$ `required_margin`. "Normalised margin" means $(R^2 - \text{control})/(1 - \text{control})$.

### 1. The models predict orbits well

Trained models beat constant-velocity extrapolation by wide margins once they are trained long enough:

- **Default seeded E1 setting:** final `val_mse_ratio` is about 0.011 [f-20261011-201533-0653].
- **$N = 16$ snapshots per orbit:** 0.0007 [f-20261013-000329-a86b].
- **6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$:** 0.0037 and 0.0032 [f-20261012-030233-ab28].

The exception is fine sampling with short training. At $N = 256$, $e \le 0.85$ and 600 steps, the model never beats the baseline (final `val_mse_ratio` 2.03–2.37) [f-20261012-034134-5e82]. At $N = 256$ and 3000 steps it gets below 1 only around step 2000, and ends at 0.24–0.32 [f-20261013-063603-95b5].

### 2. Per-snapshot acceleration is decodable; physical force is weaker and less reliable

Every evidence run cited in this section decodes `accel_step` above its control. It peaks mostly at layer 4. Force is weaker and often fails its control.

**At $N = 64$ and 3000 steps, which probe "leads" depends on the metric.**

- **Raw margin:** force leads. Margins are 0.228 / 0.255 / 0.216 for force against 0.207 / 0.178 / 0.156 for `accel_step` [f-20261013-075855-f75f]. The same holds at $L = 2$, $d_{\text{model}} = 32$ (force 0.221 / 0.255 vs `accel_step` 0.146 / 0.128) [f-20261013-000220-ab2d] and on data seed 1 [f-20261013-054512-1b08].
- **Normalised margin:** `accel_step` leads (0.68–0.74 against about 0.43–0.48 for force) [f-20261013-075855-f75f, f-20261013-054512-1b08].
- **Why:** the `accel_step` control is high, about 0.70–0.79. So raw-margin comparisons favour force.
- **Widening $m_2$:** this did not push force behind `accel_step` on raw margin (refuted) [f-20261011-203628-5cd9].

**In short runs, force does not beat its control.**

- **Single-control runs:** in 300-step $N = 64$ runs, final force margins were 0.000–0.062 [f-20261012-002713-8d93].
- **Best-of-5 controls:** final force margins were 0.033–0.080 [f-20261012-044142-d490].
- **Early transient:** every short run shows a force spike at step 10 that survives best-of-5 controls. The margin is 0.166–0.189 and decays to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7]. The same transient appears at $N = 16$ (0.106–0.127) [f-20261012-015955-fc2b].

**With longer training, force forms late.**

- **Data seed 1, 1500 steps:** force holds its margin from steps 400 / 450 / 600, while `accel_step` holds from steps 100–150 [f-20261013-025812-b9ee].
- **$d_{\text{model}} = 128$, 5000 steps:** force forms at steps 400 / 350, against step 100 for `accel_step`. After step 500 it gains +0.074 / +0.081, against +0.025 / +0.022 for `accel_step` [f-20261013-020028-8792].
- **Caveat:** the force signal usually peaks at layer 1, where `distance` is decoded. None of these runs has regressed `distance` out of the force target, so it may be a nonlinear read of distance.

### 3. When structure forms

**`distance` and `energy` form first,** typically at steps 5–50. At that point `val_mse_ratio` is still far above 1 [f-20261011-203659-4e2f, f-20261012-005950-42a7, f-20261012-044142-d490].

**At $N = 64$, `accel_step` forms before the model beats the baseline:**

- **25-step checkpoints:** the margin first reaches 0.1 at step 100 in both seeds. At that point `val_mse_ratio` is 1.96 / 1.84, and it first drops below 1 at step 175 [f-20261011-203659-4e2f].
- **10-step checkpoints:** formation comes at steps 140 / 140 / 100, against `val_mse_ratio` < 1 at steps 180 / 180 / 190 [f-20261012-002713-8d93].
- **Best-of-5 controls:** formation comes at steps 85–90, against `val_mse_ratio` < 1 at steps 135–165 [f-20261012-005950-42a7]. In other runs it comes at steps 90 / 100 / 100 [f-20261012-044142-d490].
- **Caveat:** the margins at crossing are thin. A 0.15 bar was never reached in 2 of 3 seeds (refuted) [f-20261012-005950-42a7].

![accel_step formation, N = 64, seed 4](figure:formation/run-20261011-202258-3cf9)

**"Formation before the baseline is beaten" depends on the baseline, not on the model.**

- **$N = 16$:** `val_mse_ratio` falls below 1 at step 20, but `accel_step` forms at steps 75–90. That is within 15 steps of the seed-matched $N = 64$ runs [f-20261012-015955-fc2b].
- **$N = 256$ at $e \le 0.85$:** `accel_step` forms at steps 60 / 70 / 60, matching $N = 64$, even though the model never beats the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].

**Formation is tied to training step rather than to a fixed loss fraction or to learning rate.**

- **Halving the learning rate** moved formation only from steps 60 / 60 / 70 to 70 / 80 / 100. The predicted proportional shift was refuted [f-20261012-044258-ab88].
- **A pre-registered narrow $N = 64$ window** failed: only 2 of 5 seeds landed inside it. Persistent formation spanned steps 65–90 (refuted) [f-20261012-054929-0067].

**Linear `accel_step` decodability stops tracking prediction quality.** In the 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$:

- `accel_step` $R^2$ levelled off at about 0.70–0.74. After step 2000 it gained only +0.025 / +0.022.
- Over the same stretch, `val_mse_ratio` fell 6.8× / 5.5× [f-20261012-030233-ab28].

### 4. Snapshots per orbit ($N$) affect force but not `accel_step`

Final best-layer force margin by $N$:

| Setting | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | 0.215 / 0.261 [f-20261013-015914-3f49] | — | 0.113 / 0.003✗ (rerun) [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094✗ [f-20261013-051807-f3f1] | 0.040✗ / 0.033✗ [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079✗ [f-20261013-075824-35f0] | 0.026✗ / 0.045✗ / 0.029✗ [f-20261013-075749-3d2c] |
| Data seed 0, seeds 20/21, 12000 steps | — | 0.255 / 0.266 [f-20261013-051738-2a37] | 0.193 / 0.197 [f-20261013-051738-2a37] | — |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072✗ / 0.039✗ [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.275 / 0.268 (trajectory) [f-20261013-224908-8b46] | 0.163 / 0.173 [f-20261013-224908-8b46] | — |

Runs are 3000 steps unless noted. ✗ marks a force margin below its required margin.

**$N = 16$ to $N = 64$:** no decline. The gradual-decline hypothesis was refuted [f-20261013-015914-3f49].

**$N = 64$ to $N = 128$:** the trained model loses force; the controls do not change.

- **Data seed 0:** trained force $R^2$ falls by 0.098–0.128, while the controls move by +0.002 to +0.009 [f-20261013-075824-35f0].
- **Data seed 1:** trained force $R^2$ falls by 0.064–0.072 [f-20261013-054542-23e1] and 0.095–0.097 [f-20261013-224908-8b46].
- **`distance`:** its $R^2$ does not fall over the same step [f-20261013-075824-35f0, f-20261013-224908-8b46].
- **Training length:** the gap persists at 12000 steps [f-20261013-051738-2a37].

![Force probe heatmap, N = 128, seed 24](figure:heatmap/run-20261013-073201-7611/force)

**$N = 128$ to $N = 256$:** the cause depends on the dataset.

- **Data seed 1:** the drop is entirely a control effect. Trained force $R^2$ stays at about 0.67 while the controls rise to 0.60–0.63 [f-20261013-063631-c4ed].
- **Data seed 0:** trained $R^2$ also falls, and the controls explain only about 35–50% of the loss. The controls-only hypothesis was refuted [f-20261013-075749-3d2c].

**The $N = 256$ shortfall is not just slow convergence.**

- **More steps:** 1000 extra steps added only about 0.011–0.014 of trained force $R^2$ (refuted) [f-20261013-083108-2efd].
- **Matched loss:** at matched `val_mse_ratio`, $N = 128$ still decodes force better, at 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

**`accel_step` stays roughly flat across $N$.** Its normalised margin is:

- **$N = 16$:** 0.68–0.69. A predicted rise to about 0.8 was refuted [f-20261013-000329-a86b].
- **$N = 64$:** 0.68–0.74 [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08].
- **$N = 128$:** 0.63–0.72 [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46].
- **$N = 256$:** 0.66–0.75 [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c].

### 5. Fixing the mass ratio at $m_2 = 0.5$

**The first test was refuted.** Fixing $m_2$, so that $|F| = 0.5/r^2$ exactly, did not lift force $R^2$ to 0.85. Force reached 0.750 / 0.732, against `accel_step` at 0.924 / 0.946 [f-20261011-223000-b5f9]. The budget hard stop cut both runs short:

- **Seed 14** stopped at step 1500 (completion 0.55). That is below the 80% needed to count toward replication.
- **Seed 15** stopped at step 2500 (completion 0.917).

**In complete runs, fixing $m_2$ raises the force controls to about 0.55–0.64, and the $N = 64 \to 128$ deficit survives.** Trained force $R^2$:

| Setting | $N = 64$ | $N = 128$ | Gap |
|---|---|---|---|
| Data seed 0, seeds 20/21, 3000 steps | 0.763 / 0.789 [f-20261013-234439-d851] | 0.693✗ / 0.684 [f-20261013-234552-f8e8] | 0.070 / 0.105 |
| Data seed 0, seeds 22/23, 3000 steps | ≈0.767 / 0.770 (trajectory) | 0.654✗ / 0.631✗ [f-20261014-004426-68d6] | ≈0.11 / 0.14 |
| Data seed 1, seeds 20/21, 3000 steps | 0.795 / 0.790 [f-20261014-015644-a80b] | 0.730✗ / 0.715✗ [f-20261014-015607-930a] | ≈0.065 / 0.088 |
| Data seed 0, seeds 22/23, 6000 steps | 0.8235 / 0.7684 [f-20261014-040037-f9bb] | 0.681 / 0.696 (references) | 0.143 / 0.072 |
| Data seed 0, seeds 22/23, 12000 steps | 0.8398 / 0.7935 [f-20261014-051311-3e2e] | 0.6894✗ / 0.7197 [f-20261014-050044-cd2f] | 0.151 / 0.074 |
| Data seed 1, seeds 22/23, 6000 steps | 0.866 / 0.847 [f-20261014-060043-9a03] | 0.780 / 0.794 [f-20261014-060114-ee01] | 0.086 / 0.053 |
| $d_{\text{model}} = 128$, data seed 0, seeds 22/23, 6000 steps | 0.82669 / 0.81015 [f-20261016-012922-fd3b] | 0.71829 / 0.7653 (references) | 0.108 / 0.045 |

✗ marks a force margin below its required margin.

**What the table shows:**

- **The deficit is not in the $m_2$ factor.** The hypothesis that fixing $m_2$ removes the drop was refuted [f-20261013-234552-f8e8]. Controls stay nearly flat, so what is lost at $N = 128$ appears to be the readout of $1/r^2$.
- **It persists with training.** At $N = 128$, force gained only 0.008 / 0.029 between steps 6000 and 12000, while `val_mse_ratio` fell about 6–10× [f-20261014-050044-cd2f].
- **It holds at matched loss.** The matched-loss gaps are about 0.13 and 0.07 [f-20261014-051311-3e2e].
- **It holds on data seed 1, with smaller gaps.** These are same-layer comparisons, and they keep the same seed ranking [f-20261014-060114-ee01].
- **It narrows but does not close at $d_{\text{model}} = 128$.** The gaps are 0.108 / 0.045, against 0.143 / 0.072 at $d_{\text{model}} = 64$. `accel_step`'s normalised margin is 0.730 / 0.733 [f-20261016-012922-fd3b].

![Force probe heatmap, d_model = 128, N = 64, m₂ fixed, seed 22](figure:heatmap/run-20261016-002356-cbfe/force)

**Caveats:**

- **Cross-layer comparisons.** Several pairings compare different layers, including both $d_{\text{model}} = 128$ pairs and seed 22 on data seed 0.
- **Borderline gap.** Seed 23's gap at $d_{\text{model}} = 128$ is only 0.005 above the 0.04 bar [f-20261016-012922-fd3b].
- **Possible slow convergence.** Seed 23's same-layer gap on data seed 0 narrowed from about 0.096 to 0.074 between 6000 and 12000 steps. So slower convergence is not ruled out for that seed [f-20261014-051311-3e2e].
- **Wide seed spread.** At 6000 steps on data seed 0, per-seed gaps span 0.011–0.143 [f-20261014-040037-f9bb].

### 6. Eccentricity sets the control level and caps linear decodability

`accel_step` at $N = 64$, 600 steps:

| Training $e$ range | Control $R^2$ | Trained $R^2$ | Normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |
| $\le 0.9$ [f-20261016-012952-f2b5] | 0.150–0.170 | 0.3927–0.4361 | 0.284–0.320 |

**Wider eccentricity lowers both the control and linear decodability, and the decline steepens.** Above $e \le 0.75$ the drop accelerates [f-20261013-005923-526c]. Even at $e \le 0.9$, `accel_step` still beats its control by 0.240–0.266 and forms at step 100 [f-20261016-012952-f2b5].

**The $e \le 0.9$ shortfall is a ceiling, not slow convergence.** In 3000-step runs on seeds 70–72:

- **Layer-4 `accel_step` $R^2$** settles at 0.457 / 0.421 / 0.447. That is 0.116 / 0.137 / 0.141 below matched $e \le 0.85$ values of 0.573 / 0.558 / 0.588.
- **Normalised margin** is 0.360 / 0.319 / 0.334.
- **Gain from extra training:** $R^2$ rose only 0.011–0.064 from the 600-step values. From step 1000–2250 onward it stayed flat within about 0.01, while `val_mse_ratio` fell about 10×, from about 0.12 to 0.013–0.015 [f-20261016-033018-5b87].
- **Caveat:** the $e \le 0.85$ reference values come from a separate experiment.

![accel_step formation, e ≤ 0.9, 3000 steps, seed 70](figure:formation/run-20261016-030349-a5e9)

**Force is weaker at every eccentricity ceiling.**

- **600-step runs:** normalised margins are about 0.09–0.22, force forms late (steps 250–590), and it fails its control in some seeds [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c, f-20261016-012952-f2b5].
- **3000-step runs at $e \le 0.9$:** force beats its control in all 3 seeds (margins 0.140 / 0.131 / 0.212). But its best layer varies by seed (layers 4, 1 and 2), while `accel_step` peaks at layer 4 in every seed [f-20261016-033018-5b87].

### 7. Out-of-distribution transfer

**Models trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$): `val_mse_ratio` is 0.057 / 0.055, about 3.6× in-distribution.
- **High eccentricity** ($e \in [0.5, 0.8]$): `val_mse_ratio` is 0.270 / 0.249, about 16–18× in-distribution. That fails the pre-registered 10× criterion (refuted). The models still beat the baseline, and best force $R^2$ falls to 0.40 / 0.34.
- **Three-body sets:** ratios of 72 and 70 are exploratory only.

**Training on $e \in [0, 0.8]$ closes the high-$e$ prediction gap,** to 1.05× and 1.01× in-distribution [f-20261011-211551-d718].

- **Early stop:** the budget hard stop ended both runs at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- **Probes:** force $R^2$ on held-out high-$e$ systems is still only 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### 8. Symbolic regression has not recovered $r^{-2}$

- **Strong `accel_step` probes** ($R^2$ 0.831 / 0.881): the fitted direction has the form $a^2/r - c\,a$ with $c \approx 0.55$, not the true law $a^3/r^2$. Fit errors are 0.245 / 0.201, and force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Training at $e \le 0.9$:** the fits are again $r^{-1}$ (refuted). Here the `accel_step` probe explains only 40–50% of variance [f-20261011-223027-8fb3].
- **Possible fitter artefact:** the fitter has not yet been run on the true `accel_step` target, so the $r^{-1}$ form may come from the fitter rather than the model.

### 9. Reproducibility

**The original $N = 256$ reference margins did not reproduce on an exact rerun (refuted).** The rerun gave force margins of 0.113 / 0.003 and `accel_step` margins of 0.162 / 0.198 [f-20261013-050513-2c7a].

**The current pipeline is deterministic for a fixed seed.** A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. So the force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance. Why the original runs differ is still unexplained.

### 10. Null and negative results

The 13 refuted hypotheses:

1. **Widening $m_2$ would put force behind `accel_step` on raw margin.** It did not [f-20261011-203628-5cd9].
2. **$e \le 0.5$ training would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2$ would lift force $R^2$ to 0.85.** It did not [f-20261011-223000-b5f9].
4. **Wide-$e$ training would yield an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 margin, and the early force spike would vanish under stricter controls.** Neither happened [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$.** $N = 64$ matched $N = 16$ [f-20261013-015914-3f49].
10. **An exact rerun would reproduce the original $N = 256$ margins.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128 \to 256$ force drop is entirely a control effect.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 extra steps would close the $N = 256$ gap.** They did not [f-20261013-083108-2efd].
13. **Fixing $m_2$ would remove the $N = 64 \to 128$ drop.** The drop persisted [f-20261013-234552-f8e8].

Several supported findings are also nulls for force, in that force fails its control:

- **Short $N = 64$ runs** [f-20261012-002713-8d93, f-20261012-044142-d490].
- **Most $N = 256$ runs** [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed].
- **$m_2$-fixed $N = 128$ runs** [f-20261014-004426-68d6, f-20261014-015607-930a, f-20261014-050044-cd2f].
- **One of three seeds at $e \le 0.9$, 600 steps** [f-20261016-012952-f2b5].

### 11. Not yet established

No finding yet reports:

- `velocity` probe results;
- activation patching or any other causal test;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target.

Limits that apply throughout:

- **Single controls.** The earliest findings used a single random control [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f].
- **Rounded values.** Several paired comparisons use 3-decimal trajectory values whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261014-004426-68d6, f-20261014-015607-930a, f-20261014-060114-ee01].
- **Training length.** Force was still rising at the end of most runs.
- **Datasets and architecture.** Only two data seeds have been tested, and most results come from the 4-layer, $d_{\text{model}} = 64$ architecture.
