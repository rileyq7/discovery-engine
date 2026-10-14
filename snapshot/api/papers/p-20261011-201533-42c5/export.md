# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained to predict two-body orbits (REBOUND, $G = M = 1$) encode the gravitational law or quantities specific to the task, and when that structure forms. We fit linear probes on each layer for physical force $m_2/r^2$, per-snapshot acceleration `accel_step` $\propto a^3/r^2$, `distance` and `energy`. A probe counts only if it beats the best of several random-init controls by a required margin. The lab has 49 replicated findings (36 supported, 13 refuted) from 115 evidence runs, 4 of them cut short by the budget stop. `accel_step` is decodable in every setting. Its headroom-normalised margin stays at about 0.63–0.80 across $N = 16$–$256$ snapshots per orbit [f-20261013-000329-a86b, f-20261013-075749-3d2c, f-20261014-051311-3e2e]. At $N = 64$ it forms at steps 100–140, 40–90 steps before the model beats constant-velocity extrapolation [f-20261012-002713-8d93]. Force is weaker and depends on sampling. Its margin is 0.22–0.26 at $N = 64$, falls to 0.08–0.14 at $N = 128$, and fails its control in most $N = 256$ runs [f-20261013-075855-f75f, f-20261013-075824-35f0, f-20261013-063603-95b5]. The $N = 64 \to 128$ gap survives fixing $m_2$ and 12000 training steps [f-20261014-051311-3e2e]. The main negative result is that symbolic fits to the probe directions give $r^{-1}$, not $r^{-2}$ [f-20261011-231118-7a37, f-20261011-223027-8fb3]. A fitter artefact is not yet excluded.

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

The lab's records contain 49 citable findings, all with status `replicated`. Of these, 36 are supported and 13 are refuted. Together they rest on 115 evidence runs. The budget hard stop cut 4 of those runs short, and we flag each one where its numbers are used. Unless stated otherwise, models have 4 layers and $d_\text{model} = 64$, all probes are linear, and a probe counts as beating its control only if $R^2 - \text{control } R^2 \ge \max(0.1, \text{control spread})$. Where we use the headroom-normalised margin, it is $(R^2 - \text{control})/(1 - \text{control})$. This puts `accel_step`, whose random controls are high, on a fair footing with `force`.

### 1. The models predict well, and probe structure is present

Every trained configuration below $N = 256$ beats constant-velocity extrapolation:

- **Default $e \le 0.5$ runs** end at `val_mse_ratio` ≈ 0.011 [f-20261011-201533-0653].
- **3000-step $N = 64$ runs** end at 0.014–0.018 [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08].
- **The 6-layer, $d_\text{model} = 128$ model** reaches 0.0032–0.0037 [f-20261012-030233-ab28].

`distance`, `energy` and `accel_step` beat their random controls in every completed run reported here. `force` is the only one of these targets that often fails its control (Section 3).

### 2. When structure forms

**Gravity-relevant structure forms early, in a fixed window of training steps.**

- **Single controls, seeds 2–3.** `accel_step` already beats its control at the first checkpoint (step 250), with margins 0.158 / 0.127. That is 1250 steps before `val_mse_ratio` first drops below 0.1 [f-20261011-201533-0653].
- **Denser 25-step checkpoints, seeds 4–5.** `accel_step` crosses the 0.1 margin at step 100. At that point `val_mse_ratio` is still 1.96 / 1.84, and it first falls below 1 at step 175 [f-20261011-203659-4e2f].
- **10-step checkpoints, three further seeds at $N = 64$.** Crossings come at steps 140, 140 and 100, which is 40–90 steps before the model beats the baseline [f-20261012-002713-8d93].
- **Best-of-5 controls.** Crossings come at steps 85–90 in all three seeds [f-20261012-005950-42a7].
- **Seeds 62–64.** Crossings come at steps 90–100, with the baseline beaten at steps 140–160 [f-20261012-044142-d490].

In every case `distance` and `energy` form first (steps 0–40), then `accel_step`, while `force` forms late or not at all.

![Formation trajectory at N = 64 with 10-step checkpoints, seed 46](figure:formation/run-20261011-234250-a7ca)

**"Forms before beating the baseline" is a property of the baseline, not of formation.**

- At $N = 16$, `val_mse_ratio` drops below 1 by step 20. `accel_step` still forms at steps 75–90, within 5–15 steps of the seed-matched $N = 64$ runs, so the ordering flips while the formation step barely moves [f-20261012-015955-fc2b].
- At $N = 256$ with $e \le 0.85$, `accel_step` forms at steps 60–70, the same window as $N = 64$ [f-20261012-034018-5526]. Yet these models never beat constant-velocity extrapolation (final ratio 2.03–2.37) [f-20261012-034134-5e82].
- Halving the learning rate delays formation by only +10, +20 and +30 steps, not the roughly 2× that cumulative $\text{lr} \times \text{steps}$ would predict (refuted) [f-20261012-044258-ab88].

**The formation window is less precise than first claimed.**

- A pre-registered window for $N = 64$ was met by only 2 of 5 seeds. With 5-step checkpoints, persistent formation came at steps 65–90 (refuted) [f-20261012-054929-0067].
- In 2 of 3 seeds the `accel_step` margin never reached a 0.15 bar (refuted) [f-20261012-005950-42a7].

**A brief early force signal appears and then disappears.** At $N = 64$ and $N = 16$, `force` reaches a margin of 0.106–0.189 at step 10 and decays to about 0 by steps 20–50. This survives best-of-5 controls [f-20261012-005950-42a7, f-20261012-015955-fc2b, f-20261012-044142-d490].

**A lasting force code forms only with longer training.**

- Training to 1500 steps on data seed 1 gives persistent force formation at steps 400–600, with final margins of 0.167–0.196. The best layer for force moves to layer 1, the `distance` layer [f-20261013-025812-b9ee].
- At $d_\text{model} = 128$ over 5000 steps, force forms 250–300 steps after `accel_step` and keeps strengthening late (+0.074 / +0.081 from step 500 to 5000). Over the same span `accel_step` gains only +0.025 / +0.022 [f-20261013-020028-8792].

### 3. Per-snapshot acceleration vs physical force

**Short runs at $N = 64$ hold an `accel_step` code but no `force` code.** `force` fails its control in all 3 seeds of each of two experiments (final margins 0.000–0.080) [f-20261012-002713-8d93, f-20261012-044142-d490].

**After full-length training both targets beat their controls, and which one leads depends on the metric.**

- **Raw margin favours `force`** in several settings:
  - 3000-step runs at $N = 64$ [f-20261013-075855-f75f];
  - a smaller 2-layer, $d_\text{model} = 32$ model (force 0.221 / 0.255 vs `accel_step` 0.146 / 0.128) [f-20261013-000220-ab2d];
  - a wider $m_2 \in [0.01, 1]$ range. Here the hypothesis that force would fall behind `accel_step` was refuted [f-20261011-203628-5cd9].
- **Headroom-normalised margin favours `accel_step`** in almost every case: about 0.65–0.79, against about 0.35–0.59 for `force` [f-20261011-203628-5cd9, f-20261013-054512-1b08, f-20261013-075855-f75f, f-20261014-040037-f9bb]. The one exception is seed 2 of the 2-layer model [f-20261013-000220-ab2d].

The `force` signal usually peaks at layer 1, where `distance` is decoded at $R^2$ ≈ 0.85–0.93. No finding has yet regressed `distance` out of the force target.

![accel_step probe heatmap, N = 64, seed 24](figure:heatmap/run-20261013-074422-d66a/accel_step)

**Linear decodability stops tracking prediction quality in a larger model.** In the 6-layer, $d_\text{model} = 128$ model at $e \le 0.85$, linear `accel_step` $R^2$ levels off at about 0.70–0.74 after step 2000. Over the same steps, `val_mse_ratio` falls a further 5.5–6.8× [f-20261012-030233-ab28].

### 4. Force decodability falls as sampling gets finer

The table gives the final best-layer force margin at 3000 steps unless noted. ✗ marks a margin that does not beat its control.

| | $N = 16$ | $N = 64$ | $N = 128$ | $N = 256$ |
|---|---|---|---|---|
| Data seed 0, seeds 20/21 | 0.236 / 0.275 [f-20261013-000329-a86b] | — | — | 0.113 / 0.003✗ [f-20261013-050513-2c7a] |
| Data seed 0, seeds 22/23 | — | 0.247 / 0.234 [f-20261013-015914-3f49] | 0.108 / 0.094✗ [f-20261013-051807-f3f1] | 0.040✗ / 0.033✗ [f-20261013-063603-95b5] |
| Data seed 0, seeds 24/25/26 | — | 0.228 / 0.255 / 0.216 [f-20261013-075855-f75f] | 0.128 / 0.143 / 0.079✗ [f-20261013-075824-35f0] | 0.026✗ / 0.045✗ / 0.029✗ [f-20261013-075749-3d2c] |
| Data seed 0, seeds 20/21, 12000 steps | — | 0.255 / 0.266 [f-20261013-051738-2a37] | 0.193 / 0.197 [f-20261013-051738-2a37] | — |
| Data seed 1, seeds 20/21 | — | 0.209 / 0.209 [f-20261013-054512-1b08] | 0.137 / 0.145 [f-20261013-054542-23e1] | 0.072✗ / 0.039✗ [f-20261013-063631-c4ed] |
| Data seed 1, seeds 22/23 | — | 0.268 / 0.275 (trajectory) [f-20261013-224908-8b46] | 0.173 / 0.163 [f-20261013-224908-8b46] | — |

**$N = 16$ and $N = 64$ give the same force level.** The hypothesis that force declines gradually from $N = 16$ upward was refuted [f-20261013-015914-3f49].

**From $N = 64$ to $N = 128$, force is lost in the trained model, not in the controls.**

- Data seed 0, seeds 24–26: trained force $R^2$ falls by 0.098–0.128, while controls move by only +0.002 to +0.009 [f-20261013-075824-35f0].
- Data seed 1: trained force $R^2$ falls by 0.064–0.072 for seeds 20/21 [f-20261013-054542-23e1] and by 0.095–0.097 for seeds 22/23 [f-20261013-224908-8b46].
- `distance` $R^2$ does not fall over the same step [f-20261013-075824-35f0, f-20261013-224908-8b46].

![Force probe heatmap, N = 128, seed 24](figure:heatmap/run-20261013-073201-7611/force)

**From $N = 128$ to $N = 256$, the cause depends on the dataset.**

- On data seed 1, the drop is entirely a control effect. Trained force $R^2$ stays at about 0.66–0.67, while the force controls rise to 0.60–0.63 [f-20261013-063631-c4ed].
- On data seed 0, trained $R^2$ also falls, and the controls account for only about 35–50% of the margin loss. The controls-only hypothesis was refuted [f-20261013-075749-3d2c].

**The $N = 256$ shortfall is not just slower convergence.**

- 1000 extra steps added only about 0.011–0.014 of trained force $R^2$ (refuted) [f-20261013-083108-2efd].
- Matched on `val_mse_ratio`, $N = 128$ still decodes force better: 0.641 vs 0.557 and 0.630 vs 0.569 [f-20261013-093210-2e03].

**The $N = 64$ vs $N = 128$ gap persists at 12000 steps**, at 0.062 / 0.069 in force margin [f-20261013-051738-2a37].

**`accel_step` stays roughly flat across $N$.** Its normalised margin is:

- 0.68–0.69 at $N = 16$ [f-20261013-000329-a86b]. The prediction that it would rise to about 0.8 at $N = 16$ was refuted.
- 0.68–0.74 at $N = 64$ [f-20261013-015914-3f49, f-20261013-075855-f75f, f-20261013-054512-1b08].
- 0.63–0.72 at $N = 128$ [f-20261013-051807-f3f1, f-20261013-075824-35f0, f-20261013-054542-23e1, f-20261013-224908-8b46].
- 0.66–0.75 at $N = 256$ [f-20261013-063603-95b5, f-20261013-063631-c4ed, f-20261013-075749-3d2c].

### 5. Fixing the mass ratio at $m_2 = 0.5$

**The first test was refuted.** Fixing $m_2$ did not lift force $R^2$ to 0.85: force reached 0.750 / 0.732, against `accel_step` at 0.924 / 0.946 [f-20261011-223000-b5f9]. The budget hard stop cut both runs short:

- seed 14 stopped at step 1500 (completion 0.55), below the 80% needed to count as an agreeing seed;
- seed 15 stopped at step 2500 (completion 0.917).

**Complete runs show that the $N = 64 \to 128$ force deficit survives fixing $m_2$.** Fixing $m_2$ raises the force control to about 0.55–0.64. Trained force $R^2$ in these runs:

| Setting | $N = 64$ | $N = 128$ | Gap |
|---|---|---|---|
| Data seed 0, seeds 20/21, 3000 steps | 0.763 / 0.789 [f-20261013-234439-d851] | 0.693 / 0.684 [f-20261013-234552-f8e8] | 0.070 / 0.105 |
| Data seed 0, seeds 22/23, 3000 steps | ≈0.767 / 0.770 (trajectory) | 0.654✗ / 0.631✗ [f-20261014-004426-68d6] | ≈0.11 / 0.14 |
| Data seed 1, seeds 20/21, 3000 steps | 0.795 / 0.790 [f-20261014-015644-a80b] | 0.730✗ / 0.715✗ [f-20261014-015607-930a] | ≈0.065 / 0.088 |
| Data seed 0, seeds 22/23, 6000 steps | 0.8235 / 0.7684 [f-20261014-040037-f9bb] | 0.681 / 0.696 (references) | 0.143 / 0.072 |
| Data seed 0, seeds 22/23, 12000 steps | 0.8398 / 0.7935 [f-20261014-051311-3e2e] | 0.6894✗ / 0.7197 [f-20261014-050044-cd2f] | 0.151 / 0.074 |
| Data seed 1, seeds 22/23, 6000 steps | 0.866 / 0.847 [f-20261014-060043-9a03] | 0.780 / 0.794 [f-20261014-060114-ee01] | 0.086 / 0.053 |

✗ marks a force margin below its required margin.

**What the table shows:**

- **The deficit is not in the $m_2$ factor.** The hypothesis that fixing $m_2$ removes the drop was refuted [f-20261013-234552-f8e8]. Controls stay nearly flat in every pairing, so what is lost at $N = 128$ appears to be the readout of $1/r^2$.
- **It persists through 12000 steps.** At $N = 128$, force gained only 0.008 / 0.029 between steps 6000 and 12000, while `val_mse_ratio` fell about 6–10× [f-20261014-050044-cd2f].
- **It holds when matched on loss.** Read at matched `val_mse_ratio`, the gaps are about 0.13 and 0.07 [f-20261014-051311-3e2e].
- **It holds on data seed 1, with smaller gaps.** The data-seed-1 gaps compare layer 1 with layer 1. They are smaller than on data seed 0 but keep the same seed ranking [f-20261014-060114-ee01].

**Caveats:**

- **Cross-layer comparison.** Seed 22's data-seed-0 gaps compare different layers.
- **Possible slow convergence.** Seed 23's same-layer gap on data seed 0 narrowed from about 0.096 to 0.074 between 6000 and 12000 steps, so slower convergence is not ruled out for that seed [f-20261014-051311-3e2e].
- **Wide spread across seeds.** At 6000 steps on data seed 0, per-seed gaps span 0.011–0.143 [f-20261014-040037-f9bb].
- **`accel_step` is unaffected.** Across these runs its normalised margin stays at 0.72–0.80 [f-20261014-050044-cd2f, f-20261014-051311-3e2e, f-20261014-060043-9a03].

![Formation trajectory, N = 64, m₂ fixed, 12000 steps, seed 23](figure:formation/run-20261014-043150-2de3)

### 6. Eccentricity sets the control level

`accel_step` at $N = 64$, 600 steps:

| Training $e$ range | Control $R^2$ | Trained $R^2$ | Normalised margin |
|---|---|---|---|
| $\le 0.5$ [f-20261012-044142-d490] | 0.724–0.746 | 0.890–0.902 | — |
| $\le 0.7$ [f-20261012-055051-b178] | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ [f-20261013-005923-526c] | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ [f-20261013-000359-37e2] | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ [f-20261012-034018-5526] | 0.248–0.333 | 0.507–0.594 | 0.34–0.40 |

- **Wider eccentricity lowers both the control and linear decodability**, and the decline steepens above $e \le 0.75$ [f-20261013-005923-526c].
- **Force is weaker at every eccentricity ceiling.** Its normalised margins are about 0.11–0.22, and it forms late, at steps 250–590 [f-20261012-034018-5526, f-20261012-055051-b178, f-20261013-000359-37e2, f-20261013-005923-526c].

### 7. Out-of-distribution transfer

**Trained on $e \le 0.5$** [f-20261011-203723-b7de]:

- **Low mass ratios** ($m_2 \in [0.01, 0.1]$): error is about 3.6× the in-distribution error.
- **High eccentricity** ($e \in [0.5, 0.8]$): error is 16–18× in-distribution, which fails the pre-registered 10× criterion (refuted). The models still beat the baseline, and force $R^2$ falls to 0.40 / 0.34.
- **Three-body sets:** ratios of 72 and 70 are exploratory only.

**Trained on $e \in [0, 0.8]$, the high-$e$ prediction gap closes** to 1.01–1.05× in-distribution [f-20261011-211551-d718].

- The budget hard stop ended both runs at step 2500 (completion 0.867 and 0.9), so their probes describe the step-2500 weights.
- Even with matched prediction error, force $R^2$ on held-out high-$e$ systems is only 0.366–0.377, against 0.729–0.802 on low-$e$ systems. These out-of-distribution probes have no random controls.

### 8. Symbolic regression has not recovered $r^{-2}$

- **Strong `accel_step` probes** ($R^2$ 0.831 / 0.881): the fitted direction has the form $a^2/r - c\,a$ with $c \approx 0.55$, not the true $a^3/r^2$. Fit errors are 0.245 / 0.201, and force fits as about $0.8/r$ [f-20261011-231118-7a37].
- **Training at $e \le 0.9$:** the fits are again $r^{-1}$ (refuted) [f-20261011-223027-8fb3].
- **Possible fitter artefact:** the fitter has not yet been run on the true `accel_step` target, so we cannot rule out that the $r^{-1}$ form comes from the fitter rather than the model.

### 9. Reproducibility

**The original $N = 256$ reference margins did not reproduce on an exact rerun (refuted).** The rerun gave force margins of 0.113 / 0.003 [f-20261013-050513-2c7a].

**The current pipeline is deterministic for a fixed seed.** A second identical rerun matched the first to 3 decimals at every checkpoint [f-20261013-054603-d6cc]. Two consequences follow:

- The force gap between seeds 20 and 21 at $N = 256$ is genuine seed variance.
- Why the original runs differ is still unexplained.

### 10. Null and negative results

The 13 refuted hypotheses:

1. **Widening $m_2$ would put force behind `accel_step`.** It did not [f-20261011-203628-5cd9].
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
- $m_2$-fixed $N = 128$ runs [f-20261014-004426-68d6, f-20261014-015607-930a, f-20261014-050044-cd2f].

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
- **Datasets and architecture.** Only two data seeds have been tested, and most results come from one architecture.
