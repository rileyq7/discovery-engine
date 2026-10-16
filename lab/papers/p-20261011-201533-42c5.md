# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated two-body orbits ($G = M = 1$) encode the gravitational law or a step-based surrogate, and when that structure forms. Linear probes on the residual stream target force ($m_2/r^2$), per-snapshot acceleration `accel_step` ($\propto a^3/r^2$), distance and energy. A probe result counts only if it beats several random-init controls. The lab has 51 replicated findings (38 supported, 13 refuted) from 120 evidence runs, 4 of them cut short by the budget hard stop. `accel_step` is consistently decodable. At $N = 64$ it forms at steps 85–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its normalised margin stays at roughly 0.63–0.75 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-051807-f3f1, f-20261013-063603-95b5]. Force is weaker and depends on sampling. From $N = 64$ to $N = 128$, trained force $R^2$ falls 0.098–0.128 while controls stay flat [f-20261013-075824-35f0]. This gap survives fixing $m_2$ and training for 12000 steps [f-20261013-234552-f8e8, f-20261014-051311-3e2e], and force fails its control at $N = 256$ [f-20261013-063603-95b5]. The main negative result is that symbolic fits to the decoded directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. Whether that comes from the fitter is untested.

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

The lab has 51 citable findings. All 51 are replicated: 38 support their hypothesis and 13 refute it. Together they rest on 120 evidence runs. The budget hard stop cut 4 of those runs short, and we flag each one where its numbers appear. Unless stated otherwise, models have 4 layers and $d_{\text{model}} = 64$. A probe result counts only if it clears its random-model control by the required margin, $\max(0.1, \text{control\_spread})$. The earliest findings used a single control, and we say so where they are cited.

Across these findings, four things hold. Next-step prediction beats constant-velocity extrapolation by a wide margin. The residual stream linearly encodes the per-snapshot acceleration `accel_step` $\propto a^3/((1+m_2)r^2)$ above control in every run that reports it. It forms early, at a training step that is stable across seeds, learning rates and sampling densities. Physical force $|F| = m_2/r^2$ is weaker, forms later and depends on the setup. It fades as sampling gets finer, and fixing $m_2$ does not rescue it. Symbolic fits to the probe directions have not recovered the $r^{-2}$ law.

### 1. Prediction quality

Every completed two-body model that reports a final `val_mse_ratio` beats the constant-velocity baseline, with one exception: $N = 256$ runs trained for only 600 steps, which end at 2.03–2.37 [f-20261012-034134-5e82].

- **Typical 3000–4000-step runs at $N = 64$, $e \le 0.5$:** final ratios of about 0.011–0.016 [f-20261011-201533-0653, f-20261013-015914-3f49, f-20261013-075855-f75f].
- **Longer or wider runs:** 0.0030–0.0031 at 12000 steps [f-20261013-051738-2a37], and 0.0037 / 0.0032 for a 6-layer, $d_{\text{model}} = 128$ model [f-20261012-030233-ab28].
- **Coarse sampling:** at $N = 16$ the constant-velocity baseline is weak, and the ratio reaches 0.0007 [f-20261013-000329-a86b].

### 2. When structure forms

**`accel_step` forms before the model beats the baseline at $N = 64$.**

- With 25-step checkpoints, the `accel_step` margin first reached 0.1 at step 100 in both seeds. At that point `val_mse_ratio` was still 1.96 / 1.84, and it first fell below 1 by step 175 (single control) [f-20261011-203659-4e2f].
- With 10-step checkpoints and best-of-several controls, formation came at steps 140 / 140 / 100. That is 40–90 steps before `val_mse_ratio` < 1 (steps 180–190) [f-20261012-002713-8d93].
- A second set of seeds formed at steps 85–90, 45–75 steps before the baseline crossing [f-20261012-005950-42a7].
- Persistent formation at steps 90–100 came 40–60 steps before the baseline crossing [f-20261012-044142-d490].
- In the earliest 250-step-resolution runs, gravity-related probes already beat control at step 250. That is 1250 steps before `val_mse_ratio` first fell below 0.1 (single control) [f-20261011-201533-0653].

![Probe formation, N = 64, e ≤ 0.5, seed 64](figure:formation/run-20261012-043022-144a)

**Formation follows the training step, not the loss.**

- **$N = 16$:** `val_mse_ratio` falls below 1 at step 20, but `accel_step` forms at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs. The ordering against the baseline flips while the formation step barely moves. "Beats the baseline" is therefore a baseline-dependent marker [f-20261012-015955-fc2b].
- **$e \le 0.85$:** `accel_step` forms at steps 60–70 at $N = 64$ [f-20261012-034018-5526]. At $N = 256$ it forms at steps 60–70 too, even though those models never beat the baseline [f-20261012-034134-5e82].
- **Learning rate:** halving it delays formation by only +10 / +20 / +30 steps. Proportional scaling would predict roughly 2× (refuted) [f-20261012-044258-ab88].
- **A narrow window was too narrow:** a pre-registered $N = 64$ window held for only 2 of 5 seeds. Persistent formation spanned steps 65–90, and part of the shift may come from the finer 5-step checkpoints (refuted) [f-20261012-054929-0067].

**Order of formation.** `distance` and `energy` form first (steps 5–50 at fine resolution). `accel_step` follows. Force forms late or not at all [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-044142-d490].

**The early force transient survives best-of-5 controls.** It reaches a force margin of 0.166–0.189 at step 10 and decays to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7]. A transient of 0.106–0.127 at step 10 also appears at $N = 16$ [f-20261012-015955-fc2b]. The same finding refuted the claim that `accel_step` would clear a 0.15 margin: it plateaued at 0.134–0.135 in 2 of 3 seeds.

**Force can form later with longer training.**

- On data seed 1, 1500-step runs make force beat control persistently in all 3 seeds, at steps 400–600. This happens only after force's best layer moves to layer 1, the `distance` layer. At step 600, force was already at 0.104–0.138, compared with 0.085–0.099 in 600-step runs [f-20261013-025812-b9ee].
- At $d_{\text{model}} = 128$, force forms at steps 350 / 400, against step 100 for `accel_step`. It keeps gaining through step 5000 (+0.074 / +0.081, against +0.025 / +0.022 for `accel_step`) [f-20261013-020028-8792].

### 3. `accel_step` versus force

**Raw margin and headroom-normalised margin rank the two probes differently.** `accel_step` controls are high (often 0.71–0.81 at $e \le 0.5$), which caps its raw margin. Force often has the larger raw margin, but `accel_step` leads on headroom-normalised margin, $(R^2 - c)/(1 - c)$.

- **Wider $m_2$ range:** widening $m_2$ to $[0.01, 1]$ did not put force behind `accel_step` on raw margin (refuted). On normalised margin, `accel_step` leads 0.70 vs 0.48 and 0.71 vs 0.42 [f-20261011-203628-5cd9].
- **Same pattern elsewhere:** the identical split appears at $N = 16$ [f-20261013-000329-a86b], at $N = 64$ on both data seeds [f-20261013-015914-3f49, f-20261013-054512-1b08, f-20261013-075855-f75f], and in a 2-layer, $d_{\text{model}} = 32$ cell [f-20261013-000220-ab2d].
- **Linear decodability can stop tracking loss.** For the 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$:
  - `accel_step` $R^2$ plateaus at about 0.70–0.74 and rises only +0.025 / +0.022 after step 2000, while `val_mse_ratio` falls a further 6.8× / 5.5×.
  - Its normalised margin is 0.66 / 0.61, against 0.41 for force [f-20261012-030233-ab28].

### 4. Force decodability falls as sampling gets finer

The table gives the final best-layer force margin. ✗ marks a margin that does not beat its control.

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003✗ [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094✗ [f-20261013-051807-f3f1] | 0.040✗ / 0.033✗ [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079✗ [f-20261013-075824-35f0] | 0.026✗ / 0.045✗ / 0.029✗ [f-20261013-075749-3d2c] |
| Data seed 0, seeds 20/21, 12000 steps | — | 0.255 / 0.266 [f-20261013-051738-2a37] | 0.193 / 0.197 [f-20261013-051738-2a37] | — |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072✗ / 0.039✗ [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.275 / 0.268 (trajectory) [f-20261013-224908-8b46] | 0.163 / 0.173 [f-20261013-224908-8b46] | — |

Runs are 3000 steps unless noted.

**$N = 16$ to $N = 64$:** no decline. The gradual-decline hypothesis was refuted [f-20261013-015914-3f49].

**$N = 64$ to $N = 128$:** the trained model loses force; the controls do not change.

- On data seed 0, trained force $R^2$ falls by 0.098–0.128, while controls move by +0.002 to +0.009 [f-20261013-075824-35f0].
- On data seed 1, trained force $R^2$ falls by 0.064–0.072 [f-20261013-054542-23e1] and 0.095–0.097 [f-20261013-224908-8b46].
- `distance` $R^2$ does not fall over the same step [f-20261013-075824-35f0, f-20261013-224908-8b46].
- The gap persists at 12000 steps [f-20261013-051738-2a37].

![Force probe heatmap, N = 128, seed 24](figure:heatmap/run-20261013-073201-7611/force)

**$N = 128$ to $N = 256$:** the cause depends on the dataset.

- On data seed 1, the drop is entirely a control effect [f-20261013-063631-c4ed].
- On data seed 0, trained $R^2$ also falls, and controls explain only about 35–50% of the loss. The controls-only hypothesis was refuted [f-20261013-075749-3d2c].

**The $N = 256$ shortfall is not just slow convergence.**

- 1000 extra steps added only about 0.011–0.014 of trained force $R^2$ (refuted) [f-20261013-083108-2efd].
- At matched `val_mse_ratio`, $N = 128$ still decodes force better: 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

**`accel_step` stays roughly flat across $N$.** Its normalised margin is:

- 0.68–0.69 at $N = 16$. A predicted rise to about 0.8 was refuted [f-20261013-000329-a86b].
- 0.68–0.74 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08].
- 0.63–0.72 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46].
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c].

### 5. Fixing the mass ratio at $m_2 = 0.5$

**The first test was refuted.** Fixing $m_2$ did not lift force $R^2$ to 0.85: force reached 0.750 / 0.732, against `accel_step` at 0.924 / 0.946 [f-20261011-223000-b5f9]. The budget hard stop cut both runs short:

- seed 14 stopped at step 1500 (completion 0.55), below the 80% needed to count toward replication;
- seed 15 stopped at step 2500 (completion 0.917).

**In complete runs, fixing $m_2$ raises force controls to about 0.55–0.64, and the $N = 64 \to 128$ deficit survives.** Trained force $R^2$:

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
- **It narrows but does not close at $d_{\text{model}} = 128$.** The gaps are 0.108 / 0.045, against 0.143 / 0.072 at $d_{\text{model}} = 64$. Both pairs compare different layers, and seed 23's gap is only 0.005 above the 0.04 bar. `accel_step`'s normalised margin is 0.730 / 0.733 [f-20261016-012922-fd3b].

![Force probe heatmap, d_model = 128, N = 64, m₂ fixed, seed 22](figure:heatmap/run-20261016-002356-cbfe/force)

**Caveats:**

- **Cross-layer comparisons.** Several pairings compare different layers, notably seed 22 on data seed 0.
- **Possible slow convergence.** Seed 23's same-layer gap on data seed 0 narrowed from about 0.096 to 0.074 between 6000 and 12000 steps, so slower convergence is not ruled out for that seed [f-20261014-051311-3e2e].
- **Wide seed spread.** At 6000 steps on data seed 0, per-seed gaps span 0.011–0.143 [f-20261014-040037-f9bb].

### 6. Eccentricity sets the control level

`accel_step` at $N = 64$, 600 steps:

| Training $e$ range | Control $R^2$ | Trained $R^2$ | Normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |
| $\le 0.9$ [f-20261016-012952-f2b5] | 0.150–0.170 | 0.3927–0.4361 | 0.284–0.320 |

**Wider eccentricity lowers both the control and linear decodability, and the decline steepens.**

- Above $e \le 0.75$ the drop accelerates [f-20261013-005923-526c].
- From $e \le 0.85$ to $e \le 0.9$, raw $R^2$ fell about 0.13–0.16 on seeds 70–72, relative to $e \le 0.85$ values of 0.557 / 0.541 / 0.570 from a separate experiment.
- Even at $e \le 0.9$, `accel_step` still beats its control by 0.240–0.266 and forms at step 100 [f-20261016-012952-f2b5].

These 600-step runs had not converged.

![accel_step probe heatmap, e ≤ 0.9, seed 70](figure:heatmap/run-20261016-005901-b03a/accel_step)

**Force is weaker at every eccentricity ceiling.** Its normalised margins are about 0.09–0.22, it forms late (steps 250–590), and it fails its control in some seeds [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c, f-20261016-012952-f2b5].

### 7. Out-of-distribution transfer

**Models trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$): error is about 3.6× the in-distribution error.
- **High eccentricity** ($e \in [0.5, 0.8]$): error is 16–18× in-distribution, failing the pre-registered 10× criterion (refuted). The models still beat the baseline, and best force $R^2$ falls to 0.40 / 0.34.
- **Three-body sets:** ratios of 72 and 70 are exploratory only.

**Training on $e \in [0, 0.8]$ closes the high-$e$ prediction gap** to 1.01–1.05× in-distribution [f-20261011-211551-d718].

- The budget hard stop ended both runs at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- Force $R^2$ on held-out high-$e$ systems is still only 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### 8. Symbolic regression has not recovered $r^{-2}$

- **Strong `accel_step` probes** ($R^2$ 0.831 / 0.881): the fitted direction has the form $a^2/r - c\,a$ with $c \approx 0.55$, not $a^3/r^2$. Fit errors are 0.245 / 0.201, and force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Training at $e \le 0.9$:** the fits are again $r^{-1}$ (refuted) [f-20261011-223027-8fb3].
- **Possible fitter artefact:** the fitter has not yet been run on the true `accel_step` target, so the $r^{-1}$ form may come from the fitter rather than the model.

### 9. Reproducibility

**The original $N = 256$ reference margins did not reproduce on an exact rerun (refuted).** The rerun gave force margins of 0.113 / 0.003 [f-20261013-050513-2c7a].

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

- short $N = 64$ runs [f-20261012-002713-8d93, f-20261012-044142-d490];
- most $N = 256$ runs [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed];
- $m_2$-fixed $N = 128$ runs [f-20261014-004426-68d6, f-20261014-015607-930a, f-20261014-050044-cd2f];
- one of three seeds at $e \le 0.9$ [f-20261016-012952-f2b5].

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
