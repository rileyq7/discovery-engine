# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated orbits (REBOUND, $G = M = 1$) encode the physical force $|F| = m_2/r^2$ or a shortcut, and when any structure forms. We fit linear probes to the residual stream and compare each against randomly initialised controls. We compare force with `accel_step` $\propto a^3/r^2$, the per-snapshot acceleration that next-step prediction requires. The lab has 45 replicated findings (32 supported, 13 refuted) from 107 evidence runs. `accel_step` is consistently decodable. At $N = 64$ snapshots per orbit it forms by steps 85–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its headroom-normalised margin stays near 0.63–0.79 across $N$ [f-20261013-000329-a86b, f-20261013-063603-95b5]. Force does not hold up: its margin falls from 0.209–0.275 at $N = 64$ to 0.003–0.113 at $N = 256$ [f-20261013-050513-2c7a, f-20261013-075749-3d2c]. From $N = 64$ to $N = 128$ the loss is in trained $R^2$, not in the controls [f-20261013-075824-35f0, f-20261013-224908-8b46], even with $m_2$ fixed [f-20261013-234552-f8e8]. The main negative result is that symbolic regression on the probe directions returns $r^{-1}$ forms such as $a^2/r - c\,a$ ($c \approx 0.55$), never the true $r^{-2}$ law [f-20261011-231118-7a37, f-20261011-223027-8fb3]. The fitter itself has not yet been checked on the true target.

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

The lab has 45 citable findings, all with status `replicated`. Of these, 32 support their hypothesis and 13 refute it. They rest on 107 evidence runs. The budget hard stop cut 4 of those runs short, and we flag each one below where its numbers are used. Unless stated otherwise, models have 4 layers and $d_\text{model} = 64$, use $N = 64$ snapshots per orbit and $e \le 0.5$, and are probed with linear (ridge) probes against best-of-$k$ random-init controls. Throughout, we report two kinds of margin. The *raw margin* is $R^2 - R^2_\text{control}$. The *normalised margin* is $(R^2 - R^2_\text{control}) / (1 - R^2_\text{control})$.

### Prediction quality

The transformers predict orbits far better than constant-velocity extrapolation.

- **Default runs ($N = 64$).** Final `val_mse_ratio` is about 0.011 [f-20261011-201533-0653], and 0.014–0.016 after 3000 steps [f-20261013-075855-f75f].
- **Coarse sampling.** At $N = 16$ it falls to 0.0007, because the baseline is weaker there [f-20261013-000329-a86b].
- **Larger model.** With 6 layers and $d_\text{model} = 128$ it reaches 0.0037 / 0.0032 [f-20261012-030233-ab28].
- **Fine sampling.** $N = 256$ is the exception. After 600 steps at $e \le 0.85$ it never beats the baseline (2.03–2.37) [f-20261012-034134-5e82]. At $e \le 0.5$ it drops below 1 only around step 2000 and ends at 0.24–0.39 after 3000 steps [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c].

### What is decodable: per-snapshot acceleration, not physical force

Two codes are reliably decodable in every configuration tested:

- a per-snapshot acceleration code, `accel_step` $\propto a^3/((1+m_2)r^2)$, which is best decoded at the last layer;
- a `distance` code at layer 1.

The physical force $|F| = m_2/r^2$ is weaker, and whether it is decodable depends on the configuration.

**Short runs at $N = 64$ (300 steps).** Force fails its control at the end in every seed. Final margins are 0.000–0.062 [f-20261012-002713-8d93] and 0.033–0.080 [f-20261012-044142-d490]. `accel_step` beats its control, at R² 0.884–0.905 against controls of 0.72–0.79.

**Longer training.** Force eventually forms, but late and mostly at the `distance` layer:

- **1500 steps, data seed 1.** Force first holds its 0.1 margin at steps 400 / 450 / 600, ending at 0.196 / 0.193 / 0.167. It crosses the bar only after its best layer moves from 4 to 1. Layer-4 force never exceeds about 0.085 [f-20261013-025812-b9ee].
- **$d_\text{model} = 128$, 5000 steps.** Force forms at steps 400 / 350, against step 100 for `accel_step`. From step 500 to 5000, force gains +0.074 / +0.081 while `accel_step` gains only +0.025 / +0.022 [f-20261013-020028-8792].
- **3000 steps, $N = 64$.** Force margins are 0.215–0.261 on data seed 0 [f-20261013-015914-3f49, f-20261013-075855-f75f] and 0.209 on data seed 1 [f-20261013-054512-1b08].
- **Two layers, $d_\text{model} = 32$.** Force margins are 0.221 / 0.255 [f-20261013-000220-ab2d].

**Which code leads depends on the metric.**

- On raw margin, force usually leads after long training.
- On normalised margin, `accel_step` leads in almost every case. For example, it is 0.69–0.71 against force's 0.43–0.48 [f-20261013-075855-f75f], and 0.86 / 0.83 against 0.49 / 0.53 at $d_\text{model} = 128$ [f-20261013-020028-8792].
- The one exception is $L = 2$, $d_\text{model} = 32$, seed 2, where force leads on both measures [f-20261013-000220-ab2d].

The reason for the split is that `accel_step`'s random control is high (about 0.65–0.81 at $e \le 0.5$), which caps its raw margin.

**Early-run findings (single controls).** In the earliest runs, force also kept a larger raw margin than `accel_step` when $m_2$ was widened to $[0.01, 1]$; the hypothesis that it would fall behind was refuted [f-20261011-203628-5cd9].

![accel_step probe heatmap, N = 64, seed 24](figure:heatmap/run-20261013-074422-d66a/accel_step)

### When structure forms

**`accel_step` forms early, at a nearly fixed training step.**

| Setting | `accel_step` formation step | Model first beats the baseline | Source |
|---|---|---|---|
| 250-step checkpoints | 250 (margins 0.158 / 0.127) | `val_mse_ratio` < 0.1 at step 1500 | [f-20261011-201533-0653] |
| 25-step checkpoints | 100 (margins 0.115 / 0.108) | `val_mse_ratio` < 1 first reported at step 175 | [f-20261011-203659-4e2f] |
| 10-step checkpoints, $N = 64$ | 140 / 140 / 100 | step 180–190 | [f-20261012-002713-8d93] |
| Best-of-5 controls | 85–90 | step 135–165 | [f-20261012-005950-42a7] |
| Seeds 62–64 | 90–100 | step 140–160 | [f-20261012-044142-d490] |

The two earliest rows used a single random control.

**`distance` and `energy` form earlier still**, at steps 5–40, while the model is far worse than the baseline [f-20261011-203659-4e2f, f-20261012-005950-42a7, f-20261012-044142-d490].

**Formation timing depends little on sampling density or learning rate.**

- **$N = 16$.** `accel_step` forms at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs. The model now beats the baseline first, at step 20, because the baseline is weaker at coarse sampling. So "forms before beating the baseline" is a baseline-dependent marker [f-20261012-015955-fc2b].
- **$N = 256$ at $e \le 0.85$.** Formation (steps 60–70) matches $N = 64$ (60–70) even though the $N = 256$ models never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].
- **Halving the learning rate.** Formation moves from 60 / 60 / 70 to 70 / 80 / 100. The hypothesis that formation scales in proportion to the learning rate was refuted [f-20261012-044258-ab88].
- **5-step checkpoints.** Persistent formation spans steps 65–90. A pre-registered $N = 64$ window held in only 2 of 5 seeds (refuted) [f-20261012-054929-0067].

**Formation timing is sensitive to the threshold.**

- `accel_step` margins at $e \le 0.5$ level off just above the 0.1 bar.
- Under best-of-5 controls, a 0.15 bar was not reached in 2 of 3 seeds (refuted) [f-20261012-005950-42a7].
- At $N = 16$, the 0.15 bar was never reached [f-20261012-015955-fc2b].

**Force shows an early transient that does not last.**

- A margin of 0.106–0.19 appears around steps 10–25 and collapses to about 0 by steps 20–55 [f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-015955-fc2b, f-20261012-054929-0067].
- It survives best-of-5 controls, so it is not a control artefact [f-20261012-005950-42a7].

![Probe formation, N = 64, seed 47](figure:formation/run-20261012-003058-0ea4)

### Sampling density: force decodability falls with $N$, `accel_step` does not

Final force margins after 3000 steps:

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | 0.215 / 0.261 [f-20261013-015914-3f49] | 0.150 / 0.131 [f-20261013-051807-f3f1] | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079 [f-20261013-075824-35f0] | 0.026 / 0.045 / 0.029 [f-20261013-075749-3d2c] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.268 / 0.275 (trajectory) [f-20261013-224908-8b46] | 0.173 / 0.163 [f-20261013-224908-8b46] | — |
| $m_2 = 0.5$, data seed 0, seeds 20/21 | — | 0.133 / 0.195 [f-20261013-234439-d851] | 0.063 / 0.101 [f-20261013-234552-f8e8] | — |
| $m_2 = 0.5$, data seed 0, seeds 22/23 | — | 0.193 / 0.228 [f-20261014-004505-d52c] | 0.062 / 0.080 [f-20261014-004426-68d6] | — |
| $m_2 = 0.5$, data seed 1, seeds 20/21 | — | 0.150 / 0.175 [f-20261014-015644-a80b] | 0.092 / 0.0999 [f-20261014-015607-930a] | — |

**$N = 16$ and $N = 64$ give the same force level.** The hypothesis of a gradual decline starting from $N = 16$ was refuted [f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, force falls in every paired seed, and the loss is in trained $R^2$, not in the controls.**

- **Data seed 0.** For seeds 24–26, trained force $R^2$ fell by 0.098–0.128 while controls moved by +0.002 to +0.009 [f-20261013-075824-35f0].
- **Data seed 1.** For seeds 22/23, trained $R^2$ fell by 0.095–0.097 while controls moved by −0.002 to +0.017 [f-20261013-224908-8b46].
- **Fixed $m_2$.** The drop persists with $m_2 = 0.5$; the hypothesis that fixing $m_2$ removes it was refuted. Trained $R^2$ fell by 0.070 / 0.105 [f-20261013-234552-f8e8] and by about 0.11 / 0.14 [f-20261014-004426-68d6] on data seed 0. On data seed 1 it fell by about 0.065 / 0.088, with layer-matched controls moving by only −0.007 / −0.013 [f-20261014-015607-930a].
- **Force fails its control at $N = 128$ with fixed $m_2$** in 5 of 6 such seeds [f-20261013-234552-f8e8, f-20261014-004426-68d6, f-20261014-015607-930a].
- **Interpretation.** The model loses the ability to read out $1/r^2$; the loss is not in the $m_2$ factor.
- **Long training.** After 12000 steps, $N = 64$ is still 0.062 / 0.069 above $N = 128$ [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, what drives the drop depends on the dataset.**

- **Data seed 1.** The drop is entirely a control effect: trained $R^2$ stays at about 0.66–0.67 while controls rise to 0.60–0.63 [f-20261013-063631-c4ed].
- **Data seed 0.** Trained $R^2$ also falls. Controls explain only about 35–50% of the margin loss, so the controls-only hypothesis was refuted [f-20261013-075749-3d2c].

**On data seed 0, the $N = 256$ shortfall is not just slow convergence.**

- **1000 extra steps.** These added only 0.011–0.014 of trained force $R^2$ (refuted) [f-20261013-083108-2efd].
- **Matched on `val_mse_ratio`.** $N = 128$ still decodes force better: 0.641 vs 0.557, and 0.630 vs 0.569 [f-20261013-093210-2e03].
- **No ceiling yet.** Force was still gaining 0.005–0.007 of margin per 500 steps, so a fixed ceiling has not been shown [f-20261013-083108-2efd].

![Force probe heatmap, N = 256, seed 24](figure:heatmap/run-20261013-071946-8084/force)

**The normalised `accel_step` margin stays roughly flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b]. The hypothesis that $N = 16$ would raise it was refuted.
- 0.68–0.79 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08, f-20261013-234439-d851, f-20261014-004505-d52c, f-20261014-015644-a80b].
- 0.63–0.76 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46, f-20261013-234552-f8e8, f-20261014-004426-68d6, f-20261014-015607-930a].
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c, f-20261013-083108-2efd].

### Fixing the mass ratio

**Fixing $m_2 = 0.5$ does not lift force to the hypothesised $R^2 \ge 0.85$ (refuted).** In early runs, force reached 0.750 / 0.732 while `accel_step` reached 0.924 / 0.946 [f-20261011-223000-b5f9]. The budget stop cut both of those runs short:

- seed 14 at step 1500, completion 0.55, which is below the 80% needed to count toward replication;
- seed 15 at step 2500, completion 0.917.

**In complete 3000-step runs at $N = 64$**, trained force $R^2$ is:

- 0.763–0.789 across four seeds on data seed 0 [f-20261013-234439-d851, f-20261014-004505-d52c];
- 0.795 / 0.790 on data seed 1 [f-20261014-015644-a80b].

Trained $R^2$ is higher than in wide-$m_2$ runs, but the margins are smaller, because fixing $m_2$ raises the force control to about 0.54–0.64.

### Eccentricity sets the control level

Widening the training range of $e$ lowers both the `accel_step` control and its trained $R^2$. All rows are $N = 64$:

| $e$ range | `accel_step` control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

- **The decline steepens above $e \le 0.75$** [f-20261013-005923-526c].
- **Force is weaker at every ceiling.** Its normalised margins are 0.11–0.22, and it forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].
- **Larger model at $e \le 0.85$** (6 layers, $d_\text{model} = 128$). Linear `accel_step` $R^2$ levels off at about 0.70–0.74, gaining only 0.022–0.025 after step 2000. Over the same steps, `val_mse_ratio` keeps falling 5.5–6.8× [f-20261012-030233-ab28]. So linear decodability of `accel_step` stops tracking prediction quality.

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$). Error is about 3.6× the in-distribution error, and force $R^2$ holds at 0.75–0.77.
- **High eccentricity** ($e \in [0.5, 0.8]$). Error is 16–18× the in-distribution error, which fails the pre-registered 10× criterion (refuted). The models still beat the baseline, and force $R^2$ falls to 0.40 / 0.34.
- **Three-body sets.** Ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$** [f-20261011-211551-d718]:

- **The high-$e$ gap closes.** High-$e$ error falls to 1.01–1.05× the in-distribution error.
- **Truncated runs.** The budget stop cut both runs short at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- **The probes still differ.** With prediction error matched, force $R^2$ on held-out high-$e$ systems is 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **Strong probes** (`accel_step` $R^2$ 0.831 / 0.881). The fitted direction has the form $a^2/r - c\,a$ with $c \approx 0.55$, not $a^3/r^2$. Fit errors are 0.245 / 0.201. Force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Weaker probes, training at $e \le 0.9$** (`accel_step` $R^2$ 0.40–0.50). These also fit as $r^{-1}$; the expected $r^{-2}$ did not appear (refuted) [f-20261011-223027-8fb3].
- **Possible fitter artefact.** The fitter has not been run on the true `accel_step` target, so we cannot yet rule out that it produces the $r^{-1}$ form on its own.

### Reproducibility

**The original $N = 256$ reference runs did not reproduce.** An exact rerun gave force margins of 0.113 / 0.003. At step 3000 it agreed with 6000- and 12000-step runs to within about 0.025, which rules out the learning-rate schedule horizon as the cause (refuted) [f-20261013-050513-2c7a].

**The current pipeline is deterministic for a fixed seed.** A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. It follows that:

- the 0.11 force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance;
- the cause of the original discrepancy is still unexplained.

### Null and negative results

The 13 refuted hypotheses are:

1. **A wider $m_2$ range would put force behind `accel_step`.** It did not [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2$ would lift force $R^2$ to 0.85.** It did not [f-20261011-223000-b5f9].
4. **Wide-$e$ training would yield an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar, and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$.** $N = 64$ matched $N = 16$ [f-20261013-015914-3f49].
10. **An exact rerun would reproduce the original $N = 256$ margins.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128 \to 256$ force drop is all control effect.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 extra steps would close the $N = 256$ gap.** They did not [f-20261013-083108-2efd].
13. **Fixing $m_2$ would remove the $N = 64 \to 128$ drop.** The drop persisted [f-20261013-234552-f8e8].

Several supported findings are also nulls for force, where force fails its control:

- short $N = 64$ runs [f-20261012-002713-8d93, f-20261012-044142-d490];
- $N = 256$ in most seeds [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed];
- $m_2$-fixed runs at $N = 128$ [f-20261014-004426-68d6, f-20261014-015607-930a].

### Not yet tested, and limits

The lab has not yet reported:

- `velocity` probe results;
- activation patching, or any other causal test of whether the decoded codes are used;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target;
- matched-loss comparisons for data seed 1 and for the $m_2$-fixed arms.

Several limits apply throughout:

- **Probes.** All probes are linear.
- **Single controls in early runs.** The earliest findings used a single random control [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f].
- **Formation steps.** `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-000329-a86b, f-20261013-054512-1b08].
- **Rounded values.** Several paired comparisons rely on 3-decimal trajectory values whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261014-004426-68d6, f-20261014-015607-930a].
- **Training length.** Force was still rising at the end of most 3000-step runs.
- **Datasets.** Only two data seeds have been tested.
- **Architecture.** Most results come from one architecture (4 layers, $d_\text{model} = 64$).
