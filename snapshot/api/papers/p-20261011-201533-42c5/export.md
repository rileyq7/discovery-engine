# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict simulated two-body orbits learn the physical law $|F| = m_2/r^2$ or a task-specific substitute, and when any such structure forms. We fit linear probes to the residual stream and count a result only if it beats the best of several randomly initialised models by the required margin. We compare physical force against the per-snapshot acceleration $(F/m_2)\,dt^2$ (`accel_step`) that next-step prediction needs. The lab has 43 replicated findings (30 supported, 13 refuted) from 103 evidence runs; the budget stop cut 4 of those runs short. In every run cited here, `accel_step` beats its control, and in the runs with fine checkpoints it forms within 60–140 training steps, before the model beats constant-velocity extrapolation [f-20261012-002713-8d93, f-20261012-034018-5526]. Its headroom-normalised margin stays roughly flat (0.63–0.79) from $N = 16$ to $N = 256$ snapshots per orbit [f-20261013-000329-a86b, f-20261013-075824-35f0, f-20261013-063603-95b5]. Force is weaker and fragile. Its margin falls from 0.216–0.255 at $N = 64$ to 0.026–0.045 at $N = 256$ [f-20261013-075855-f75f, f-20261013-075749-3d2c], and the drop persists with $m_2$ fixed [f-20261013-234552-f8e8]. The main negative result is that symbolic fits to the decoded direction give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3].

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

The lab has 43 citable findings, all with replicated status. Of these, 30 support their hypothesis and 13 refute it. They rest on 103 evidence runs. Four of those runs were cut short by the budget hard stop, and we flag them wherever their numbers appear. In all experiments the models beat constant-velocity extrapolation on held-out systems, with one exception: 600-step runs at $N = 256$.

The main result is that the residual stream holds the per-snapshot acceleration that next-step prediction needs, $|a_{\text{step}}| = (2\pi/N)^2 a^3 / ((1+m_2) r^2)$. Probes decode it above random controls in every setting tested, and its headroom-normalised margin barely changes with sampling density. Physical force, $|F| = m_2/r^2$, behaves differently. It forms later, it is absent from short runs, it usually peaks in the same layer as `distance`, and its decodability falls as $N$ grows above 64. Symbolic fits to the decoded directions have not recovered the $r^{-2}$ law.

### Prediction quality

- **Default runs.** The seeded E1 configuration reached `val_mse_ratio` ≈ 0.011 in both seeds [f-20261011-201533-0653].
- **Other settings.**
  - At $N = 16$, `val_mse_ratio` was 0.0007 [f-20261013-000329-a86b].
  - With $d_\text{model} = 128$ it was 0.0037 / 0.0032 at 6 layers [f-20261012-030233-ab28] and 0.0059 / 0.0053 at 4 layers [f-20261013-020028-8792].
- **The exception.** At $N = 256$, 600-step runs never beat the baseline (final 2.03–2.37) [f-20261012-034134-5e82]. 3000-step runs did, ending at 0.24–0.32 [f-20261013-063603-95b5, f-20261013-075749-3d2c].
- **Reading the ratio across $N$.** The constant-velocity baseline gets stronger as $N$ increases, so `val_mse_ratio` cannot be compared directly across $N$ [f-20261012-015955-fc2b].

### Per-snapshot acceleration versus physical force

**Short runs at $N = 64$ (300 steps) contain an `accel_step` code but no force code.**

- `accel_step` beat its control in every seed, with final margins of 0.109–0.167 at layer 4.
- Force failed its control in every seed, with final margins of 0.000–0.080 [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490].
- In 5 further seeds, force cleared the bar in 2, but only barely (0.103 and 0.105) [f-20261012-054929-0067].

**Longer training produces a force signal, but it forms late.**

- **1500 steps, data seed 1.** Force first held above its control at steps 400–600, ending at margins of 0.167–0.196. Over the same period `accel_step` gained only 0.006–0.018 after step 600 [f-20261013-025812-b9ee].
  - Force only crossed the bar after its best layer moved to layer 1, which is where `distance` is decoded.
- **3000 steps at $N = 64$.** Final force margins were 0.209–0.261 across data seeds and model seeds [f-20261013-015914-3f49, f-20261013-054512-1b08, f-20261013-075855-f75f].

**Which probe looks stronger depends on the metric.**

- Force usually has the larger raw margin.
- `accel_step` has the larger headroom-normalised margin, $(R^2 - R^2_\text{ctrl})/(1 - R^2_\text{ctrl})$. For example, it was 0.71 / 0.69 / 0.68 against 0.43–0.48 for force [f-20261013-075855-f75f], and 0.70 against 0.48 in the earliest runs [f-20261011-203628-5cd9].
- The raw-margin lead for force mostly reflects the high `accel_step` control (0.69–0.81 at $e \le 0.5$), which caps how large its raw margin can get.
- The same raw-margin pattern holds in a smaller model ($L = 2$, $d = 32$): force 0.221 / 0.255 against `accel_step` 0.146 / 0.128 [f-20261013-000220-ab2d].

**In wider models, force strengthens late while the `accel_step` readout levels off.**

- **4 layers, $d = 128$.** Force formed persistently at steps 400 and 350, against step 100 for `accel_step`. From step 500 to 5000, force gained +0.074 / +0.081 while `accel_step` gained +0.025 / +0.022 [f-20261013-020028-8792].
- **6 layers, $d = 128$, $e \le 0.85$.** Linear `accel_step` $R^2$ levelled off at about 0.70–0.74. Over the same steps `val_mse_ratio` kept falling 6.8× and 5.5× [f-20261012-030233-ab28]. Past that point, linear decodability stops tracking prediction quality.

![accel_step probe heatmap, N = 64, m₂ = 0.5, seed 22](figure:heatmap/run-20261014-001211-4848/accel_step)

### When structure forms

**`distance` and `energy` form first.** They reach their bar within steps 5–50, while the model is still far worse than the baseline [f-20261011-203659-4e2f, f-20261012-002713-8d93, f-20261012-044142-d490].

**`accel_step` forms next, before the model beats the baseline at $N = 64$:**

| Finding | Checkpoint spacing | `accel_step` forms | `val_mse_ratio` first < 1 |
|---|---|---|---|
| [f-20261011-203659-4e2f] | 25 | step 100 (2 seeds) | step 175 |
| [f-20261012-002713-8d93] | 10 | steps 140 / 140 / 100 | steps 180 / 180 / 190 |
| [f-20261012-005950-42a7] | 5 | steps 85–90 | steps 135–165 |
| [f-20261012-044142-d490] | 10 | steps 90–100 | steps 140–160 |

Caveats on this ordering:

- **First checkpoint.** In the first E1 runs, `accel_step` already beat its control at the first checkpoint (step 250), while `val_mse_ratio` was still 0.87 [f-20261011-201533-0653].
- **Thin margins at the crossing.** The margins when `accel_step` crosses are only just above 0.1. A 0.15 bar was never reached in 2 of 3 seeds [f-20261012-005950-42a7].
- **Window too narrow.** A pre-registered formation window was refuted: across 5 seeds, persistent formation spanned steps 65–90 [f-20261012-054929-0067].

**The formation step barely moves with sampling density, so "before the baseline" is a property of the baseline.**

- **$N = 16$.** The model beats the baseline at step 20, but `accel_step` forms at steps 75–90. That is within 5–15 steps of the matching $N = 64$ seeds, so the ordering flips while the formation step stays put [f-20261012-015955-fc2b].
- **$N = 256$, $e \le 0.85$.** `accel_step` forms at steps 60–70 [f-20261012-034134-5e82], the same as $N = 64$ (60–70) [f-20261012-034018-5526], even though these $N = 256$ models never beat the baseline.

**Halving the learning rate delays formation only slightly.** The delay was +10 / +20 / +30 steps, not the roughly 2× that proportional scaling predicts, so the proportional-scaling hypothesis was refuted [f-20261012-044258-ab88].

**A transient force spike appears early and then collapses.**

- Force margins of 0.166–0.189 at step 10 survive best-of-5 controls, then decay to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7].
- The same spike appears at $N = 16$ [f-20261012-015955-fc2b] and in other $N = 64$ seeds [f-20261012-044142-d490, f-20261012-054929-0067].

![Probe formation, N = 64, seed 46 (10-step checkpoints)](figure:formation/run-20261011-234250-a7ca)

### The mass ratio $m_2$

**Widening the $m_2$ range did not make force fall behind `accel_step` on raw margin** (refuted) [f-20261011-203628-5cd9].

**Fixing $m_2$ did not lift force $R^2$ to the predicted 0.85** (refuted). The fixed-$m_2$ models reached 0.750 / 0.732 [f-20261011-223000-b5f9]. Both of these runs were cut short by the budget stop:

- Seed 14 stopped at step 1500 (completion 0.55). That is below the 80% replication threshold, so this refutation effectively rests on seed 15.
- Seed 15 stopped at step 2500 (completion 0.917).

**Complete 3000-step runs with $m_2 = 0.5$ at $N = 64$ decode force above control in all four seeds:**

| Seeds | Trained force $R^2$ | Force control | Force margin | Normalised `accel_step` margin |
|---|---|---|---|---|
| 20 / 21 [f-20261013-234439-d851] | 0.763 / 0.789 | 0.630 / 0.594 | 0.133 / 0.195 | 0.79 / 0.74 |
| 22 / 23 [f-20261014-004505-d52c] | 0.767 / 0.770 | 0.574 / 0.542 | 0.193 / 0.228 | 0.747 / 0.756 |

- **Trained $R^2$ is the stable quantity.** It spans 0.763–0.789 across the four seeds, while the control spans 0.54–0.63.
- **Seed 23's control missed its pre-registered range.** At 0.542 it falls just below [0.55, 0.68] [f-20261014-004505-d52c].
- **Higher trained $R^2$, smaller margins.** Trained force $R^2$ is higher than in the same seeds' wide-$m_2$ runs, but the margins are smaller. Fixing $m_2$ raises the force control from about 0.48–0.51 to 0.59–0.63 [f-20261013-234439-d851].

### Sampling density: force decodability falls with $N$, `accel_step` does not

Final force margins after 3000 steps:

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | 0.215 / 0.261 [f-20261013-015914-3f49] | 0.150 / 0.131 [f-20261013-051807-f3f1] | 0.113 / 0.003 (rerun) [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094 [f-20261013-051807-f3f1] | 0.040 / 0.033 [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079 [f-20261013-075824-35f0] | 0.026 / 0.045 / 0.029 [f-20261013-075749-3d2c] |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072 / 0.039 [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.268 / 0.275 (trajectory) [f-20261013-224908-8b46] | 0.173 / 0.163 [f-20261013-224908-8b46] | — |
| $m_2 = 0.5$, seeds 20/21 | — | 0.133 / 0.195 [f-20261013-234439-d851] | 0.063 / 0.101 [f-20261013-234552-f8e8] | — |
| $m_2 = 0.5$, seeds 22/23 | — | 0.193 / 0.228 [f-20261014-004505-d52c] | 0.062 / 0.080 [f-20261014-004426-68d6] | — |

**$N = 16$ and $N = 64$ give the same force level.** A gradual decline from $N = 16$ was therefore refuted [f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, force falls in every paired seed, and the loss is in the trained model rather than the controls.**

- **Data seed 0.**
  - Seeds 24–26: trained force $R^2$ fell by 0.098–0.128 while controls moved by only +0.002 to +0.009 [f-20261013-075824-35f0].
  - Seeds 22/23: the drop again sits in lower trained $R^2$ [f-20261013-051807-f3f1].
- **Data seed 1.**
  - Seeds 20/21: the margin fell by 0.064–0.072 [f-20261013-054542-23e1].
  - Seeds 22/23: trained $R^2$ fell by 0.095–0.097 while controls changed by −0.002 to +0.017 [f-20261013-224908-8b46].
- **With $m_2 = 0.5$.** The hypothesis that fixing $m_2$ removes the drop was refuted. Trained $R^2$ fell by 0.070 / 0.105 for seeds 20/21 [f-20261013-234552-f8e8] and by about 0.11 / 0.14 for seeds 22/23 [f-20261014-004426-68d6]. In all four seeds the controls moved by about 0.02 or less.
  - At $N = 128$, neither seed 22 nor seed 23 beat its control [f-20261014-004426-68d6].
  - The loss is therefore in reading out $1/r^2$, not in the $m_2$ factor.
- **What does not fall.** Layer-1 `distance` $R^2$ held or fell only slightly in these comparisons [f-20261013-075824-35f0, f-20261013-224908-8b46, f-20261013-234552-f8e8].
- **Long training.** After 12000 steps, $N = 64$ was still 0.062 / 0.069 above $N = 128$ [f-20261013-051738-2a37].

**From $N = 128$ to $N = 256$, what drives the drop depends on the dataset.**

- **Data seed 1.** The drop is entirely a control effect. Trained $R^2$ stayed at about 0.66–0.67 while controls rose to 0.60–0.63 [f-20261013-063631-c4ed].
- **Data seed 0.**
  - Seeds 24–26: trained $R^2$ also fell, and controls account for about 35–50% of the margin loss (the hypothesis that controls explain all of it was refuted) [f-20261013-075749-3d2c].
  - Seed 23: about two thirds of the margin loss comes from the control [f-20261013-063603-95b5].

**On data seed 0, the $N = 256$ shortfall is not only slower convergence.**

- **1000 more steps.** Training to 4000 steps added only 0.011–0.014 of trained force $R^2$, and force still failed its control (the hypothesis that this closes the gap was refuted) [f-20261013-083108-2efd].
- **Matched on `val_mse_ratio`.** $N = 128$ still decoded force better, 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].
- **No ceiling shown yet.** Force was still gaining 0.005–0.007 of margin per 500 steps, so a fixed ceiling has not been demonstrated [f-20261013-083108-2efd].

![Force probe heatmap, N = 128, m₂ = 0.5, seed 22](figure:heatmap/run-20261014-002233-049b/force)

**The normalised `accel_step` margin is roughly flat across $N$:**

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b];
- 0.68–0.79 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08, f-20261013-234439-d851, f-20261014-004505-d52c];
- 0.63–0.76 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46, f-20261013-234552-f8e8, f-20261014-004426-68d6];
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c, f-20261013-083108-2efd].

The hypothesis that $N = 16$ would raise this margin was refuted [f-20261013-000329-a86b].

### Eccentricity sets the control level

Widening the training range of $e$ lowers both the `accel_step` control and the trained $R^2$ (all $N = 64$):

| $e$ range | `accel_step` control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

- **The decline steepens above $e \le 0.75$** [f-20261013-005923-526c].
- **Force is weaker at every ceiling.** Its normalised margins are 0.11–0.22, and it forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$). Error was about 3.6× the in-distribution error, and force $R^2$ held at 0.75–0.77.
- **High eccentricity** ($e \in [0.5, 0.8]$). Error was 16–18× the in-distribution error, failing the pre-registered 10× criterion (refuted). The models still beat the baseline, and force $R^2$ fell to 0.40 / 0.34.
- **Three-body sets.** Ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$** [f-20261011-211551-d718]:

- **The high-$e$ gap closes.** High-$e$ error fell to 1.01–1.05× the in-distribution error.
- **Truncated runs.** Both runs were cut short by the budget stop at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- **The probes still differ.** With matched prediction error, force $R^2$ on held-out high-$e$ systems was 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **Strong probes** (`accel_step` $R^2$ 0.831 / 0.881). The fitted direction is $a^2/r - c\,a$ with $c \approx 0.55$, not $a^3/r^2$. Force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Weaker probes at $e \le 0.9$** (`accel_step` $R^2$ 0.40–0.50). These also fit as $r^{-1}$ (wide-$e$ training was expected to give $r^{-2}$; refuted) [f-20261011-223027-8fb3].
- **Possible fitter artefact.** The fitter has not been run on the true `accel_step` target, so we cannot yet rule out that the fitter itself produces the $r^{-1}$ form.

### Reproducibility

**The original $N = 256$ reference runs did not reproduce.** An exact rerun gave force margins of 0.113 / 0.003 against the original 0.033–0.040. At step 3000 the rerun agreed with the 6000- and 12000-step runs to within about 0.025, which rules out the learning-rate schedule horizon as the explanation (refuted) [f-20261013-050513-2c7a].

**The current pipeline is deterministic for a fixed seed on this machine.** A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. It follows that:

- the 0.11 force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance;
- why the original runs differed is still unexplained.

### Null and negative results

The 13 refuted hypotheses are:

1. **A wider $m_2$ range would push force behind `accel_step`.** It did not [f-20261011-203628-5cd9].
2. **Training on $e \le 0.5$ would transfer to high $e$ within 10×.** The error was 16–18× [f-20261011-203723-b7de].
3. **Fixing $m_2$ would lift force $R^2$ to 0.85.** It reached 0.750 / 0.732, in runs cut short by the budget stop [f-20261011-223000-b5f9].
4. **Wide-$e$ training would give an $r^{-2}$ fit.** The fits were $r^{-1}$ [f-20261011-223027-8fb3].
5. **`accel_step` would clear a 0.15 bar, and the early force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate.** The shift was sub-proportional [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow window.** Only 2 of 5 seeds did [f-20261012-054929-0067].
8. **$N = 16$ would raise the normalised `accel_step` margin.** It was level with $N = 64$ [f-20261013-000329-a86b].
9. **Force would decline gradually from $N = 16$.** $N = 64$ matched $N = 16$ [f-20261013-015914-3f49].
10. **An exact $N = 256$ rerun would reproduce the original margins.** It did not [f-20261013-050513-2c7a].
11. **The $N = 128 \to 256$ force drop comes from the controls alone.** On data seed 0, trained $R^2$ also fell [f-20261013-075749-3d2c].
12. **1000 extra steps at $N = 256$ would close the gap to $N = 128$.** They did not [f-20261013-083108-2efd].
13. **Fixing $m_2$ would remove the $N = 64 \to 128$ force drop.** The drop persisted [f-20261013-234552-f8e8].

Several supported findings are themselves nulls for force:

- **Short $N = 64$ runs.** Force fails its control at the end of these runs [f-20261012-002713-8d93, f-20261012-044142-d490].
- **$N = 256$.** Force fails its control in most seeds [f-20261012-034134-5e82, f-20261013-063603-95b5, f-20261013-063631-c4ed].
- **$m_2$-fixed $N = 128$, seeds 22/23.** Force fails its control in both seeds [f-20261014-004426-68d6].

### Not yet tested, and limits

The lab has not yet reported:

- `velocity` probe results;
- activation patching, or any other causal test of whether the decoded codes are used;
- a force probe with `distance` and `accel_step` regressed out;
- a random-control pool shared across $N$;
- nonlinear or log-target probes;
- symbolic regression on the true `accel_step` target;
- matched-loss comparisons for data seed 1 and for the $m_2$-fixed arms;
- an explanation of the original $N = 256$ discrepancy.

Several limits apply throughout:

- **Probes.** All probes are linear.
- **Formation steps.** `formed_at_step` sometimes lags the trajectory by one checkpoint [f-20261012-054929-0067, f-20261013-054512-1b08, f-20261013-000329-a86b].
- **Rounded values.** Several paired comparisons rely on 3-decimal trajectory values whose $R^2$ field may be a maximum over layers [f-20261013-075749-3d2c, f-20261013-234552-f8e8, f-20261014-004426-68d6].
- **Single controls in early runs.** The earliest findings used a single random control [f-20261011-201533-0653, f-20261011-203628-5cd9, f-20261011-203659-4e2f]. Their `distance` and `energy` margins should be read against the up-to-0.2 spread that random models show on those targets.
- **Datasets.** All four $m_2$-fixed seeds share data seed 0 [f-20261013-234439-d851, f-20261014-004505-d52c].
- **Architecture.** Most results come from one architecture (4 layers, $d_\text{model} = 64$).
