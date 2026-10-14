# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated two-body orbits encode the physical force $|F| = m_2/r^2$ or the per-snapshot acceleration $(F/m_2)\,dt^2$ that next-step prediction actually needs, and when either forms during training. Linear probes on the residual stream count only if they beat the best of several random-init controls by the required margin. Across 46 replicated findings (33 supported, 13 refuted; 109 evidence runs, 4 cut short by the budget hard stop), `accel_step` beats its control in every setting we tested. At $N = 64$ snapshots per orbit it forms by steps 85–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its headroom-normalised margin stays at about 0.63–0.75 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-051807-f3f1, f-20261013-063631-c4ed]. Force is decodable at $N = 64$ (margins 0.216–0.255) [f-20261013-075855-f75f]. At $N = 128$ it loses 0.098–0.128 of trained $R^2$ while the controls stay flat [f-20261013-075824-35f0]. It fails its control at $N = 256$ [f-20261013-075749-3d2c], and the drop persists with $m_2$ fixed [f-20261013-234552-f8e8]. The main negative result is that symbolic fits to the probe directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. Transfer to high eccentricity also failed, with error 16–18× in-distribution [f-20261011-203723-b7de].

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

The lab currently has 46 citable findings, all with status `replicated`. Of these, 33 are `supported` and 13 are `refuted`. Together they draw on 109 evidence runs. The budget hard stop cut 4 of those runs short, and we flag them wherever their numbers are used. Unless noted, the model has 4 layers and $d_\text{model} = 64$. All results come from linear probes. A probe "beats its control" when $R^2 - \text{control}\ R^2 \ge \max(0.1,\ \text{control spread})$.

### 1. What the residual stream encodes: per-snapshot acceleration, and force only weakly

**`accel_step` beats its random control at the final checkpoint in every run reported in the findings.** It usually peaks at layer 4. The target is
$$|a_\text{step}| = \left(\tfrac{2\pi}{N}\right)^2 \frac{a^3}{(1+m_2)\,r^2}.$$
Its random controls are high, typically 0.65–0.81 at $e \le 0.5$. This caps the raw margin, so we also report the headroom-normalised margin $(R^2 - c)/(1 - c)$, where $c$ is the control $R^2$.

**The physical force $|F| = m_2/r^2$ is much less reliable.** It fails its control in short runs, at fine sampling, and in several fixed-mass conditions (§3–§5). When it does pass, it usually peaks at layer 1, the same layer where `distance` is decoded. The two targets rank differently depending on the metric:

- **Raw margin** often favours force. Examples: 0.228 / 0.255 / 0.216 for force against 0.207 / 0.178 / 0.156 for `accel_step` [f-20261013-075855-f75f]; see also [f-20261013-000220-ab2d, f-20261013-054512-1b08].
- **Normalised margin** favours `accel_step` in every comparison reported. Examples: 0.71 / 0.69 / 0.68 for `accel_step` against about 0.43–0.48 for force [f-20261013-075855-f75f]; 0.86 / 0.83 against 0.49 / 0.53 in a $d_\text{model} = 128$ model [f-20261013-020028-8792].
- **The one exception to the normalised ordering** is the $L = 2$, $d_\text{model} = 32$ sweep cell. There, force leads on normalised margin in one of two seeds [f-20261013-000220-ab2d].

**Widening $m_2$ to $[0.01, 1]$ did not push force behind `accel_step` on raw margin (refuted).** Force margins were 0.243 / 0.235 against 0.205 / 0.175 for `accel_step` [f-20261011-203628-5cd9].

**`distance` and `energy` form very early.** They form at steps 10–50 in the short-run experiments, while the model is still far worse than the baseline [f-20261011-203659-4e2f, f-20261012-002713-8d93]. Their controls disagree by up to about 0.25, so their margins carry less weight than those for `accel_step`.

### 2. When structure forms

**`accel_step` forms before the model beats constant-velocity extrapolation at $N = 64$, but only just.**

- **250-step checkpoints, single control.** In the first test, the margin was already 0.158 / 0.127 at step 250, the first trained checkpoint. `val_mse_ratio` first fell below 0.1 only at step 1500 [f-20261011-201533-0653].
- **25-step checkpoints.** The crossing came at step 100. At that point `val_mse_ratio` was still 1.96 / 1.84, and it first fell below 1 at step 175 [f-20261011-203659-4e2f].
- **10-step checkpoints, best-of-5 controls.** Crossings came at steps 140 / 140 / 100. That is 40–90 steps before `val_mse_ratio` < 1. Force did not beat its control in any seed [f-20261012-002713-8d93].
- **Margin size.** The margins at crossing are thin (0.100–0.12).

![Formation trajectory, N = 64, seed 46](figure:formation/run-20261011-234250-a7ca)

**Two parts of the follow-up hypothesis failed (refuted)** [f-20261012-005950-42a7]:

- The `accel_step` margin never reached 0.15 in 2 of 3 seeds.
- An early force spike survived best-of-5 controls in all 3 seeds. The force margin was 0.166–0.189 at step 10, then decayed to about 0.00–0.02 by steps 40–50.

The same brief force transient appears in other short runs [f-20261012-044142-d490, f-20261012-054929-0067, f-20261012-015955-fc2b].

**The formation window at $N = 64$ is wider than first thought.**

- Seeds 62–64 fell inside the pre-registered windows. Persistent formation came at steps 90–100, and half the final margin was reached at steps 60–90 [f-20261012-044142-d490].
- Five further seeds at 5-step resolution formed at steps 65–90. Three of the five fell outside the window, so that prediction was refuted [f-20261012-054929-0067].

**Formation follows the training step, not the loss.**

- **$N = 16$.** `val_mse_ratio` drops below 1 at step 20, well before `accel_step` forms at steps 75–90. The formation steps are within 5–15 steps of the seed-matched $N = 64$ runs. The ordering against the baseline therefore flips only because the baseline is weaker at $N = 16$ [f-20261012-015955-fc2b].
- **$e \le 0.85$.** `accel_step` forms at steps 60–70 at $N = 64$ [f-20261012-034018-5526]. At $N = 256$ it also forms at steps 60–70, even though that model never beats the baseline (final `val_mse_ratio` 2.03–2.37) [f-20261012-034134-5e82].
- **Learning rate.** Halving it delayed formation by only +10 / +20 / +30 steps, not about 2×. The hypothesis that formation scales with learning rate was refuted [f-20261012-044258-ab88].

**Force forms later, and only with longer training.**

- **600-step runs at $N = 64$.** Force did not beat its control [f-20261012-002713-8d93, f-20261012-044142-d490].
- **1500-step runs on data seed 1.** Force formed persistently at steps 400–600 in all 3 seeds, ending at margins of 0.167–0.196. This happened only after its best layer moved to layer 1 [f-20261013-025812-b9ee].
- **$d_\text{model} = 128$ model.** Force formed 250–300 steps after `accel_step`. It gained +0.074 / +0.081 between steps 500 and 5000, while `accel_step` gained +0.025 / +0.022 [f-20261013-020028-8792].
- **$L = 2$, $d_\text{model} = 32$.** The force margin kept rising through step 3000, while `accel_step` levelled off at 0.13–0.15 by step 1500 [f-20261013-000220-ab2d].

### 3. Sampling density $N$: `accel_step` stays flat while force declines

**Final best-layer force margin at 3000 steps unless noted (✗ = does not beat its control):**

| Condition | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003✗ [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094✗ [f-20261013-051807-f3f1] | 0.040✗ / 0.033✗ [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079✗ [f-20261013-075824-35f0] | 0.026✗ / 0.045✗ / 0.029✗ [f-20261013-075749-3d2c] |
| Data seed 0, seeds 20/21, 12000 steps | — | 0.255 / 0.266 [f-20261013-051738-2a37] | 0.193 / 0.197 [f-20261013-051738-2a37] | — |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072✗ / 0.039✗ [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.268 / 0.275 (trajectory) [f-20261013-224908-8b46] | 0.173 / 0.163 [f-20261013-224908-8b46] | — |

**$N = 16$ and $N = 64$ give the same force level.** The hypothesis that force declines gradually starting from $N = 16$ was refuted [f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, force is lost in the trained model, not in the controls.**

- Data seed 0, seeds 24–26: trained force $R^2$ fell by 0.098–0.128, while controls moved by +0.002 to +0.009 [f-20261013-075824-35f0].
- Data seed 1, seeds 22/23: trained force $R^2$ fell by 0.095–0.097 [f-20261013-224908-8b46].
- Data seed 1, seeds 20/21: trained force $R^2$ fell by 0.064–0.072, again with flat controls [f-20261013-054542-23e1].
- `distance` $R^2$ does not fall over the same step [f-20261013-075824-35f0, f-20261013-224908-8b46].

**From $N = 128$ to $N = 256$, the cause depends on the dataset.**

- On data seed 1, the drop is entirely a control effect. Trained force $R^2$ stays at about 0.66–0.67, while the force controls rise to 0.60–0.63 [f-20261013-063631-c4ed].
- On data seed 0, trained $R^2$ also falls, and the controls account for only about 35–50% of the margin loss. The controls-only hypothesis was refuted [f-20261013-075749-3d2c].

**The $N = 256$ shortfall on data seed 0 is not just slower convergence.**

- 1000 extra steps added only about 0.011–0.014 of trained force $R^2$. The hypothesis that extra training would close the gap was refuted [f-20261013-083108-2efd].
- Matched on `val_mse_ratio`, $N = 128$ still decodes force better: 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

**At 12000 steps the $N = 64$ vs $N = 128$ gap remains**, at 0.062 / 0.069 in force margin. Force gained only 0.019 / 0.007 between steps 6000 and 12000 [f-20261013-051738-2a37].

**The normalised `accel_step` margin is roughly flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b]
- 0.68–0.74 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08]
- 0.63–0.72 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46]
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c]

The prediction that $N = 16$ would raise this margin to about 0.8 was refuted [f-20261013-000329-a86b].

![Force probe heatmap, N = 128, seed 24](figure:heatmap/run-20261013-073201-7611/force)

### 4. Fixing the mass ratio at $m_2 = 0.5$

**The first test was refuted.** Fixing $m_2$ did not lift force $R^2$ to 0.85. Force reached 0.750 / 0.732, against `accel_step` at 0.924 / 0.946 [f-20261011-223000-b5f9]. The budget hard stop cut both runs short:

- seed 14 stopped at step 1500 (completion 0.55), below the 80% needed to count as an agreeing seed;
- seed 15 stopped at step 2500 (completion 0.917).

**Complete 3000-step runs at $N = 64$:**

- Trained force $R^2$ is 0.789 / 0.763 on data seed 0 [f-20261013-234439-d851] and 0.795 / 0.790 on data seed 1 [f-20261014-015644-a80b].
- These values are higher than with a wide $m_2$ range, but the margins are smaller (0.133–0.195). Fixing $m_2$ raises the force control to about 0.59–0.64.

**Fixing $m_2$ does not remove the $N = 64 \to 128$ drop (refuted)** [f-20261013-234552-f8e8]. Trained force $R^2$ still falls with near-flat controls:

- 0.070 / 0.105 for seeds 20/21 on data seed 0 [f-20261013-234552-f8e8];
- about 0.11 / 0.14 for seeds 22/23 on data seed 0 [f-20261014-004426-68d6];
- about 0.065 / 0.088 on data seed 1 [f-20261014-015607-930a].

At $N = 128$ and 3000 steps, force fails its control in 5 of these 6 seeds. What is lost appears to be the readout of $1/r^2$, not of the $m_2$ factor.

**Longer training narrows the gap but has not closed it.**

- **6000 steps.** $N = 64$ reaches force $R^2$ 0.8235 / 0.7684. That is 0.143 / 0.072 above the $N = 128$ 6000-step references (0.681 / 0.696). The seed-22 comparison is cross-layer, and the per-seed gaps reported span 0.011–0.143 [f-20261014-040037-f9bb].
- **12000 steps.** $N = 128$ ends at 0.6894 (seed 22, layer 3) and 0.7197 (seed 23, layer 1). That is still 0.078 / 0.050 below the 3000-step $N = 64$ values [f-20261014-050044-cd2f].
  - Seed 22's force margin of 0.094 does not beat its control.
  - From step 6000 to 12000, force gained only 0.008 / 0.029, while `val_mse_ratio` fell about 6–10×.
  - Seed 22 has levelled off. Seed 23 is still rising slowly (about 0.002 per 1000 steps), so its lower ceiling is not yet shown to be a plateau.
  - `accel_step` levelled off at normalised margins of 0.773 / 0.787.

![Formation trajectory, N = 128, m₂ fixed, 12000 steps, seed 23](figure:formation/run-20261014-035310-270b)

### 5. Eccentricity sets the control level and limits linear decodability

**`accel_step` at $N = 64$, 600 steps, best-of-5 controls:**

| Training $e$ range | Control $R^2$ | Trained $R^2$ | Normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

**The decline steepens above $e \le 0.75$** [f-20261013-005923-526c].

**Force is weaker at every eccentricity ceiling.** Its normalised margins are about 0.11–0.22, and it forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

**Linear decodability stops tracking prediction quality in a larger model** (6 layers, $d_\text{model} = 128$, $e \le 0.85$). Linear `accel_step` $R^2$ levels off at about 0.70–0.74 after step 2000. Over the same steps, `val_mse_ratio` keeps falling 5.5–6.8× [f-20261012-030233-ab28].

### 6. Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$): error is about 3.6× the in-distribution error.
- **High eccentricity** ($e \in [0.5, 0.8]$): error is 16–18× in-distribution, which fails the pre-registered 10× criterion (refuted). The models still beat the baseline, and force $R^2$ falls to 0.40 / 0.34.
- **Three-body sets:** ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$, the high-$e$ gap closes** to 1.01–1.05× in-distribution [f-20261011-211551-d718].

- The budget hard stop ended both runs at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- Even with matched prediction error, force $R^2$ on held-out high-$e$ systems is only 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### 7. Symbolic regression has not recovered $r^{-2}$

- **Strong `accel_step` probes** ($R^2$ 0.831 / 0.881): the fitted direction has the form $a^2/r - c\,a$ with $c \approx 0.55$, not $a^3/r^2$. Fit errors are 0.245 / 0.201. Force fits as about $0.8/r$. The finding reports the same $r^{-1}$ form in 3 of 3 seeds [f-20261011-231118-7a37].
- **Training at $e \le 0.9$** (`accel_step` $R^2$ 0.395 / 0.503): the fits are again $r^{-1}$. The hypothesis that wider eccentricity would yield $r^{-2}$ was refuted [f-20261011-223027-8fb3].
- **Possible fitter artefact:** the fitter has not yet been run on the true `accel_step` target, so we cannot rule out that the $r^{-1}$ form comes from the fitter rather than the model.

### 8. Reproducibility

**The original $N = 256$ reference margins did not reproduce on an exact rerun (refuted).** The rerun gave force margins of 0.113 / 0.003. Its step-3000 values agree with 6000- and 12000-step runs to within about 0.025 [f-20261013-050513-2c7a].

**The current pipeline is deterministic for a fixed seed.** A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. Two consequences follow:

- The 0.11 force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance.
- Why the original runs differ is still unexplained.

### 9. Null and negative results

The 13 refuted hypotheses:

1. **Widening $m_2$ would put force behind `accel_step`.** It did not [f-20261011-203628-5cd9].
2. **$e \le 0.5$ training would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2$ would lift force $R^2$ to 0.85.** It did not; one of the two runs was below 80% completion [f-20261011-223000-b5f9].
4. **Wide-$e$ training would yield an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 margin, and the early force spike would vanish under best-of-5 controls.** Neither happened [f-20261012-005950-42a7].
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
- $N = 256$ in most seeds [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed];
- $m_2$-fixed runs at $N = 128$ [f-20261014-004426-68d6, f-20261014-015607-930a], including seed 22 after 12000 steps [f-20261014-050044-cd2f].

### 10. Not yet tested, and limits

No finding yet reports:

- `velocity` probe results;
- activation patching or any other causal test;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target.

Limits that apply throughout:

- **Probes.** All probes are linear.
- **Single controls.** The earliest findings used a single random control [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f].
- **Formation steps.** `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-000329-a86b, f-20261013-054512-1b08].
- **Rounded values.** Several paired comparisons use 3-decimal trajectory values whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261014-004426-68d6, f-20261014-015607-930a].
- **Training length.** Force was still rising at the end of most runs.
- **Datasets and architecture.** Only two data seeds have been tested, and most results come from one architecture.
