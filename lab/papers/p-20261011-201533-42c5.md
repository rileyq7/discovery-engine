# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated orbits (REBOUND, $G = M = 1$) encode the gravitational law, and when that structure forms. We fit linear probes on the residual stream and judge each against several randomly initialised controls. We probe two targets: the physical force $m_2/r^2$, and the per-snapshot acceleration $(F/m_2)\,dt^2$ that next-step prediction actually needs. The lab has 33 replicated findings: 23 supported and 10 refuted, from 80 runs, 4 of which were cut short by the budget stop. The per-snapshot code is robust. At $N = 64$ snapshots per orbit it clears its control at steps 85–140, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-005950-42a7]. Its headroom-normalised margin stays at 0.63–0.75 from $N = 16$ to $N = 256$ [f-20261013-000329-a86b, f-20261013-051807-f3f1, f-20261013-063603-95b5, f-20261013-063631-c4ed]. Force forms later, at steps 350–600 [f-20261013-020028-8792, f-20261013-025812-b9ee]. Its margin falls from 0.247/0.234 at $N = 64$ to 0.108/0.094 at $N = 128$ [f-20261013-015914-3f49, f-20261013-051807-f3f1]; beyond $N = 128$ the drop comes largely from rising controls [f-20261013-063631-c4ed]. The main negative result is that symbolic regression on the probe directions gives $r^{-1}$, not $r^{-2}$, in every seed tested [f-20261011-231118-7a37, f-20261011-223027-8fb3]. Models trained on $e \le 0.5$ also failed to transfer to high eccentricity, with errors 16–18× the in-distribution error [f-20261011-203723-b7de].

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

The lab has 33 citable findings, and all 33 have replicated. Of these, 23 support their pre-registered hypothesis and 10 refute it. They rest on 80 evidence runs. The budget hard stop cut 4 of those runs short, and we mark this wherever their numbers appear.

Every probe is linear. A probe result counts only if it beats the best of several random-initialisation controls by $\max(0.1, \text{control spread})$. When we say "margin", we mean $R^2$ minus the control $R^2$.

### Prediction quality

The models beat constant-velocity extrapolation by a wide margin across most configurations:

- final `val_mse_ratio` was about 0.011 after 4000 steps [f-20261011-201533-0653];
- 0.0007 at $N = 16$ [f-20261013-000329-a86b];
- 0.003 after 12000 steps at $N = 64$ [f-20261013-051738-2a37];
- 0.005–0.006 with $d_{\text{model}} = 128$ [f-20261013-020028-8792].

One exception: at $N = 256$ with $e \le 0.85$, 600 steps were not enough to beat the baseline (final `val_mse_ratio` 2.03–2.37) [f-20261012-034134-5e82].

### A per-snapshot acceleration code forms early and at a stable training step

**`accel_step` beats its control early, often before the model beats the baseline.**

- **First runs (250-step checkpoints).** The `accel_step` margin already exceeded the bar at step 250, at 0.158 and 0.127. That is 1250 steps before `val_mse_ratio` first fell below 0.1 [f-20261011-201533-0653].
- **25-step checkpoints.** `accel_step` formed at step 100 in both seeds. `val_mse_ratio` was still 1.96 and 1.84 at that point, and it first fell below 1 at step 175 [f-20261011-203659-4e2f].
- **10-step checkpoints.** `accel_step` formed at steps 140, 140 and 100, against `val_mse_ratio` < 1 at steps 180, 180 and 190 [f-20261012-002713-8d93].
- **Best-of-5 controls.** `accel_step` formed at steps 85–90 [f-20261012-005950-42a7] and 90–100 [f-20261012-044142-d490].
- **5-step checkpoints.** `accel_step` formed at steps 65–90 [f-20261012-054929-0067].

![Formation timeline at N = 64, seed 46: accel_step forms at step 100, before val_mse_ratio < 1](figure:formation/run-20261011-234250-a7ca)

**`distance` and `energy` form even earlier.** They cross at steps 10–25, while the model is still far worse than the baseline [f-20261011-203659-4e2f, f-20261012-005950-42a7]. This suggests geometry that a probe can read early but that does not yet help prediction.

**The formation step barely depends on sampling density or eccentricity.**

- **$N = 16$.** `accel_step` formed at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs. The coarser sampling makes the baseline weaker, so `val_mse_ratio` drops below 1 at step 20 and the "forms before beating the baseline" ordering flips [f-20261012-015955-fc2b]. A `val_mse_ratio` threshold therefore cannot time formation across different $N$.
- **$e \le 0.85$.** `accel_step` formed at steps 60, 60 and 70 at $N = 64$ [f-20261012-034018-5526], and at 60, 70 and 60 at $N = 256$ [f-20261012-034134-5e82]. The $N = 256$ models never beat the baseline at all.
- **Learning rate.** Halving the learning rate delayed formation by only +10, +20 and +30 steps, not the roughly 2× a proportional model would predict [f-20261012-044258-ab88].

**Margins near the threshold.** At $e \le 0.5$ the `accel_step` control is high (0.72–0.79), which caps the raw margin. In 2 of 3 seeds the margin never reached 0.15 [f-20261012-005950-42a7], and at $N = 16$ none of them did [f-20261012-015955-fc2b].

### Force: a late, weaker, layer-unstable code

**In short runs, force does not beat its control.** In 300-step runs at $N = 64$, the final force margins were 0.000–0.080 [f-20261012-002713-8d93, f-20261012-044142-d490]. A transient early spike survives best-of-5 controls: the force margin reached 0.166–0.189 at step 10, then decayed to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7].

**With longer training, force does form, late.**

- **1500 steps, data seed 1.** Force formed persistently at steps 400, 450 and 600, with final margins of 0.196, 0.193 and 0.167. The "no force code" null in short runs is therefore partly an artefact of where training stopped [f-20261013-025812-b9ee].
- **$d_{\text{model}} = 128$.** Force formed at steps 400 and 350, against step 100 for `accel_step` [f-20261013-020028-8792].
- **Wider eccentricity ranges.** At $e \le 0.8$ and $e \le 0.75$, force formed at steps 250–500 and was still rising at step 600 [f-20261013-000359-37e2, f-20261013-005923-526c].
- **3000 steps at $N = 64$.** Force margins reached 0.247 and 0.234 [f-20261013-015914-3f49].
- **12000 steps.** Force margins reached 0.255 and 0.266 [f-20261013-051738-2a37].
- **Early 4000-step runs.** By step 4000 the force margin (0.223, 0.230) had overtaken the `accel_step` margin (0.204, 0.172), although `accel_step` kept the higher raw $R^2$ [f-20261011-201533-0653].

![Force probe heatmap, N = 64, 12000 steps, seed 20](figure:heatmap/run-20261013-043205-0031/force)

**Which code leads depends on the metric.** Force often has the larger raw margin, because the `accel_step` control is high. Measured as a fraction of the available headroom, $(R^2 - c)/(1 - c)$, `accel_step` leads in every multi-seed comparison in the table below:

| Setting | `accel_step` | force |
|---|---|---|
| Wide $m_2$ range [f-20261011-203628-5cd9] | 0.70, 0.71 | 0.48, 0.42 |
| 6-layer model [f-20261012-030233-ab28] | 0.66, 0.61 | 0.41 |
| $N = 64$, 3000 steps [f-20261013-015914-3f49] | 0.68, 0.70 | 0.45, 0.43 |
| $N = 64$, data seed 1 [f-20261013-054512-1b08] | 0.725, 0.740 | 0.44, 0.45 |
| $N = 128$, data seed 0 [f-20261013-051807-f3f1] | 0.67, 0.63 | ≈0.20, 0.18 |
| $N = 128$, data seed 1 [f-20261013-054542-23e1] | 0.70, 0.72 | 0.29, 0.30 |
| $d_{\text{model}} = 128$, data seed 1 [f-20261013-020028-8792] | 0.86, 0.83 | 0.49, 0.53 |
| 1500 steps, data seed 1 [f-20261013-025812-b9ee] | 0.66–0.71 | 0.35–0.43 |

The 2-layer, $d_{\text{model}} = 32$ model is the exception. In 2 of 2 seeds, force has the larger raw margin (0.221 and 0.255, against 0.146 and 0.128). On normalised margin the comparison splits, one seed each way [f-20261013-000220-ab2d].

**Confound: force may partly be a read of distance.** The layer where force decodes best is unstable:

- it ranges from layer 0 to layer 4 [f-20261012-002713-8d93];
- it drifts across layers 1–6 in the 6-layer model [f-20261012-030233-ab28];
- it moved 4 → 1 → 4 during training in one seed [f-20261013-020028-8792].

In the 1500-step runs, force cleared the bar only after its best layer moved to layer 1, the `distance` layer. Layer-4 force never exceeded about 0.085 there [f-20261013-025812-b9ee]. In most long runs at $N \le 128$, force also peaks at layer 1 [f-20261013-015914-3f49, f-20261013-051738-2a37, f-20261013-051807-f3f1]. Until a residualised probe is run, the force margin should be read as force decodability, not as evidence of a separate $m_2$-weighted force representation.

**Fixing the mass ratio did not lift force.** With $m_2 = 0.5$, so that $|F| = 0.5/r^2$ exactly, force $R^2$ stayed at 0.750 and 0.732, while `distance` decoded at 0.87–0.89 [f-20261011-223000-b5f9]. The budget stop cut both runs short. Seed 14 reached only 0.55 completion, below the 80% replication bar, so the result rests mainly on seed 15 (0.917 completion).

### Sampling density

**Force margin across $N$, at 3000 steps:**

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |

**$N = 16$ and $N = 64$ give the same force level** [f-20261013-000329-a86b, f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, the drop is real.** It comes from lower trained $R^2$, not from higher controls: 0.56–0.57 against 0.68–0.70 on data seed 0 [f-20261013-051807-f3f1], and 0.661–0.662 against 0.736–0.742 on data seed 1 [f-20261013-054542-23e1]. The size of the drop depends on the dataset: about 0.14 on data seed 0, against 0.064–0.072 on data seed 1. After 12000 steps the $N = 64$ margin still sits 0.062 and 0.069 above $N = 128$, but in one seed the gap was narrowing [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, the drop is partly or wholly a control effect.**

- **Data seed 0.** Force never beat its control at any checkpoint. However, trained force $R^2$ fell by only 0.044 and 0.021, while the controls rose. In seed 23, about two thirds of the drop comes from the control [f-20261013-063603-95b5].
- **Data seed 1.** Trained force $R^2$ did not fall at all (about 0.66 at $N = 128$, 0.671–0.672 at $N = 256$). The force controls rose from 0.52 to 0.60–0.63, so the entire drop is in the control [f-20261013-063631-c4ed].
- **Timing caveat.** These $N = 256$ runs first fell below the baseline only around step 2000, so the result may reflect timing rather than a ceiling [f-20261013-063603-95b5].

![Force probe heatmap, N = 256, data seed 1, seed 21: trained R² is level with N = 128, but the control is higher](figure:heatmap/run-20261013-062708-a077/force)

**The original $N = 256$ reference did not reproduce, but the reruns are deterministic.** An exact rerun of the original $N = 256$ config (seeds 20/21) gave force margins of 0.113 and 0.003, and `accel_step` margins of 0.162 and 0.198. The original runs had 0.033–0.040 for force and 0.231–0.245 for `accel_step`.

- **Schedule horizon ruled out.** At step 3000 the rerun matches 6000- and 12000-step runs to within about 0.025 [f-20261013-050513-2c7a].
- **Second rerun identical.** It matched the first to 3 decimals at all 7 checkpoints for all four probes [f-20261013-054603-d6cc].
- **What follows.** The 0.11 force gap between seeds 20 and 21 at $N = 256$ is real seed-to-seed variance [f-20261013-054603-d6cc].
- **Open question.** Why the original runs differ is unresolved [f-20261013-050513-2c7a].

**Normalised `accel_step` margin is flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.70 at $N = 64$ [f-20261013-015914-3f49];
- 0.725–0.740 at $N = 64$, data seed 1 [f-20261013-054512-1b08];
- 0.63–0.67 at $N = 128$, data seed 0 [f-20261013-051807-f3f1];
- 0.70–0.72 at $N = 128$, data seed 1 [f-20261013-054542-23e1];
- 0.66 and 0.73 at $N = 256$, data seed 0 [f-20261013-063603-95b5];
- 0.75 and 0.69 at $N = 256$, data seed 1 [f-20261013-063631-c4ed].

So the per-snapshot acceleration code stays equally readable from $N = 16$ to $N = 256$. Linear force decodability falls from $N = 64$ to $N = 128$. Beyond $N = 128$, the apparent decline cannot be separated from rising random-control $R^2$. A control pool shared across $N$ would be needed to settle it.

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

The decline is uneven. About 0.05 of $R^2$ is lost from $e \le 0.7$ to $e \le 0.75$, then about 0.19 from $e \le 0.75$ to $e \le 0.85$ [f-20261013-005923-526c]. Force stays much weaker at every eccentricity ceiling, with normalised margins of 0.16–0.22 at $e \le 0.7$–$0.8$ [f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### Linear decodability stops tracking prediction quality

In the 6-layer, $d_{\text{model}} = 128$ model at $e \le 0.85$, layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 and from 0.683 to 0.705 after step 2000. Over the same steps, `val_mse_ratio` fell 6.8× and 5.5× [f-20261012-030233-ab28].

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- Error on high-$e$ systems was 16–18× the in-distribution error. This fails the pre-registered 10× criterion, although the model still beats the baseline.
- Transfer to low mass ratios held at about 3.6×.

**Trained on $e \in [0, 0.8]$,** the high-$e$ gap closed to about 1.05× and 1.01× [f-20261011-211551-d718]:

- The budget stop cut both runs short at step 2500 (completion 0.867 and 0.9).
- Force probe $R^2$ on held-out high-$e$ systems stayed at 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **With strong probes** ($R^2$ 0.831 and 0.881), the `accel_step` direction fits as $a^2/r - c\,a$ with $c \approx 0.55$ (fit errors 0.245 and 0.201). The force fits were $0.81/r$ and $0.80/r$ [f-20261011-231118-7a37].
- **At $e \le 0.9$, with weak probes** ($R^2$ 0.395 and 0.503), the fits were also $r^{-1}$ [f-20261011-223027-8fb3].
- **Open check:** the fitter has not been run on the true `accel_step` target, so a fitting artefact cannot be ruled out [f-20261011-231118-7a37].

### Null and negative results

Ten hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step` on margin.** Force kept the larger raw margin [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×.** The gap was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 and 0.732, and the budget stop cut both runs short [f-20261011-223000-b5f9].
4. **Wide-$e$ training would give an $r^{-2}$ symbolic fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow pre-registered window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was 0.69 and 0.68, level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$ to $N = 256$.** It was refuted because $N = 64$ matched $N = 16$ [f-20261013-015914-3f49]. That verdict relied on the original $N = 256$ runs, which did not reproduce. The later $N = 128$ and $N = 256$ arms show a decline, but beyond $N = 128$ it is partly control-driven [f-20261013-051807-f3f1, f-20261013-063603-95b5, f-20261013-063631-c4ed].
10. **An exact $N = 256$ rerun would reproduce the original margins within 0.02.** It did not [f-20261013-050513-2c7a]. A second rerun then reproduced the first exactly [f-20261013-054603-d6cc].

### Not yet tested, and limits

**Not yet reported:**

- `velocity` probe results;
- activation patching, or any other causal test of whether the codes are used;
- any force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$ [f-20261013-063631-c4ed];
- an explanation of why the original $N = 256$ runs differ [f-20261013-050513-2c7a].

**Limits throughout:**

- All probes are linear.
- `formed_at_step` sometimes lags the trajectory by one checkpoint, and the findings use the trajectory value [f-20261012-054929-0067, f-20261013-054512-1b08].
- Five findings use data seed 1 [f-20261013-020028-8792, f-20261013-025812-b9ee, f-20261013-054512-1b08, f-20261013-054542-23e1, f-20261013-063631-c4ed]. All others use data seed 0.
- Determinism has been shown for one config on one machine [f-20261013-054603-d6cc].
- Most results come from a single 4-layer, $d_{\text{model}} = 64$ architecture.
