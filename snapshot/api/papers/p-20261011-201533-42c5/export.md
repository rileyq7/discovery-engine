# Discovery Engine: running draft

## Abstract

We ask whether small transformers trained on simulated two-body orbits ($G = M = 1$) encode the physical force law $|F| = m_2/r^2$ or a substitute that is specific to the task, and when any such structure forms. We fit linear probes to the residual stream and score each one against best-of-several random-model controls. Each probe compares physical force with the per-snapshot acceleration $(F/m_2)\,dt^2$ (`accel_step`) that next-step prediction needs. The record holds 21 replicated findings (13 supported, 8 refuted) from 54 evidence runs, 4 of which were cut short by the budget stop. `accel_step` beats its control in every setting tested, usually at the last layer. At $N = 64$, force mostly fails its control, with final margins of 0.000–0.080 [f-20261012-002713-8d93, f-20261012-044142-d490]. The `accel_step` code forms in a narrow window of roughly 50–140 training steps that barely moves with $N$. Halving the learning rate delayed it by only 10–30 steps [f-20261012-002713-8d93, f-20261012-015955-fc2b, f-20261012-034134-5e82, f-20261012-044258-ab88]. Late in training, linear decodability stops tracking prediction: `accel_step` $R^2$ rose by at most 0.025 while `val_mse_ratio` fell 5.5–6.8× [f-20261012-030233-ab28]. The main negative result is that symbolic fits to the probe direction give $r^{-1}$, not $r^{-2}$, in all four seeds fitted [f-20261011-223027-8fb3, f-20261011-231118-7a37]. An artefact of the fitter has not yet been ruled out.

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

The lab has 21 citable findings, all replicated across seeds. Of these, 13 are supported and 8 are refuted. They rest on 54 evidence runs. The budget hard stop cut 4 of those runs short, and we flag them wherever their numbers are used. The evidence for the earliest findings lists no `control_spread`, which means those runs were judged against a single random control. Later runs use best-of-3 or best-of-5 controls.

### Per-snapshot acceleration is decodable in every run; physical force depends on the run

The `accel_step` probe beat its random-model control in every evidence run. Its target is the per-snapshot acceleration $(F/m_2)\,dt^2 = (2\pi/N)^2 a^3/((1+m_2) r^2)$.

The physical-force probe, $|F| = m_2/r^2$, behaved differently:

- In runs of 2500–4000 steps it usually beat its control, sometimes by a larger raw margin than `accel_step`.
- In 300–600-step runs with dense checkpoints it usually failed.

Table 1 gives final-checkpoint margins ($R^2$ minus control $R^2$). Where a finding does not state the margin, the table gives $R^2$ against control instead.

| Condition | `accel_step` margin | force margin | force beats control |
|---|---|---|---|
| Default, 4000 steps [f-20261011-201533-0653] | 0.204, 0.172 | 0.223, 0.230 | 2 of 2 |
| $m_2 \in [0.01, 1]$ [f-20261011-203628-5cd9] | 0.205, 0.175 | 0.243, 0.235 | 2 of 2 |
| 25-step checkpoints [f-20261011-203659-4e2f] | 0.156, 0.165 | 0.068, 0.103 | 1 of 2 |
| Transfer, in-distribution [f-20261011-203723-b7de] | 0.178, 0.208 | 0.209, 0.166 | 2 of 2 |
| $e \in [0, 0.8]$, stopped early [f-20261011-211551-d718] | 0.337, 0.358 | 0.208, 0.312 | 2 of 2 |
| $m_2 = 0.5$, stopped early [f-20261011-223000-b5f9] | 0.18, 0.18 | 0.19, 0.224 | 2 of 2 |
| $e \in [0, 0.9]$ [f-20261011-223027-8fb3] | 0.220, 0.321 | 0.097, 0.106 | 1 of 2 |
| Symbolic replication [f-20261011-231118-7a37] | 0.344, 0.299 | $R^2$ 0.692 vs 0.451; 0.660 vs 0.395 | 2 of 2 |
| 6 layers, $d = 128$, $e \in [0, 0.85]$ [f-20261012-030233-ab28] | 0.506, 0.465 | 0.300, 0.333 | 2 of 2 |
| $L = 2$, $d = 32$, 1000 systems [f-20261013-000220-ab2d] | 0.146, 0.128 | 0.221, 0.255 | 2 of 2 |
| $N = 16$, 3000 steps [f-20261013-000329-a86b] | 0.131, 0.140 | 0.236, 0.275 | 2 of 2 |
| $N = 64$, 10-step checkpoints [f-20261012-002713-8d93] | 0.109, 0.119, 0.158 | 0.040, 0.000, 0.062 | 0 of 3 |
| $N = 64$, best-of-5 controls [f-20261012-005950-42a7] | 0.134–0.135 in 2 seeds | 0.039–0.059 | 0 of 3 |
| $N = 64$, seeds 62–64 [f-20261012-044142-d490] | 0.144–0.167 | 0.033–0.080 | 0 of 3 |
| $N = 64$, seeds 65–69 [f-20261012-054929-0067] | 0.130–0.158 | 0.103, 0.105; 0.047–0.090 in the other three | 2 of 5 |
| $N = 16$, 300 steps [f-20261012-015955-fc2b] | 0.117–0.144 | 0.030, 0.055, 0.123 | 1 of 3 |
| $N = 64$, $e \in [0, 0.7]$ [f-20261012-055051-b178] | 0.257, 0.260, 0.241 | 0.140, 0.101, 0.121 | 3 of 3 |
| $N = 64$, $e \in [0, 0.8]$ [f-20261013-000359-37e2] | levels off at 0.275–0.298 | 0.124–0.150 | 3 of 3 |
| $N = 64$, $e \in [0, 0.85]$ [f-20261012-034018-5526] | 0.261, 0.287, 0.258 | 0.094–0.112 | 1 of 3 |
| $N = 64$, $e \in [0, 0.85]$, lr 0.0005 [f-20261012-044258-ab88] | 0.244–0.257 | 0.065–0.078 | 0 of 3 |
| $N = 256$, $e \in [0, 0.85]$ [f-20261012-034134-5e82] | 0.268, 0.210, 0.226 | 0.004–0.031 | 0 of 3 |

*Table 1. Final-checkpoint probe margins over random controls. The "stopped early" rows were cut short by the budget hard stop. In the $m_2 = 0.5$ row, seed 14 reached only 0.55 completion.*

**Raw margin understates `accel_step`.** The `accel_step` random controls are often high, which caps its raw margin. A headroom-normalised margin, $(R^2 - \text{control})/(1 - \text{control})$, corrects for this:

- With $m_2 \in [0.01, 1]$, force had the larger raw margin, but on the normalised margin `accel_step` led: 0.70 vs 0.48 and 0.71 vs 0.42 [f-20261011-203628-5cd9].
- In the $L = 2$, $d = 32$ cell, force led on raw margin in all three seeds (counting seed 0). On the normalised margin the ordering split:
  - `accel_step` led in seed 1 (0.46 vs 0.36).
  - Force led in seed 2 (0.42 vs 0.37) [f-20261013-000220-ab2d].
- In the 6-layer model, force's normalised margin was 0.41 in both seeds, against 0.66 and 0.61 for `accel_step` [f-20261012-030233-ab28].
- At $N = 16$ with 3000 steps, the `accel_step` control reached 0.80–0.81, the highest the lab has seen [f-20261013-000329-a86b].

**Force signal in short runs.**

- At $N = 64$ with seeds 62–64, force never beat its control, while `accel_step` $R^2$ reached 0.890–0.902 against controls of 0.724–0.746 [f-20261012-044142-d490].
- The two seeds in the 65–69 replication that did clear the bar did so only just: margins of 0.103 and 0.105 against a 0.1 bar, with control spread 0.034–0.048 [f-20261012-054929-0067].

**Early force transient.** A brief force signal appears early in training and then fades:

- 0.166–0.189 at step 10 under best-of-5 controls, decaying to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7];
- 0.148–0.159 at step 10 in seeds 62–64 [f-20261012-044142-d490];
- 0.106–0.127 at $N = 16$ [f-20261012-015955-fc2b];
- 0.12–0.19 at steps 10–25 in seeds 65–69, falling to 0.002–0.10 by steps 40–55 [f-20261012-054929-0067].

**Force over longer training.**

- In the $L = 2$ cell, the force margin kept rising from step 500 to step 3000: 0.112 to 0.221, and 0.136 to 0.255. The `accel_step` margin levelled off at 0.13–0.15 by step 1500 [f-20261013-000220-ab2d].
- At $e \in [0, 0.8]$, force formed late (steps 250–400) and was still rising at step 600 [f-20261013-000359-37e2].

**Possible confound.** Force and `accel_step` are collinear through $1/r^2$, so part of the force signal may be distance information [f-20261011-203628-5cd9]. In the $L = 2$ cell, force peaks at layer 1, the same layer where distance is decoded at $R^2$ 0.84–0.85 [f-20261013-000220-ab2d]. No run has yet regressed distance out of the force target.

![accel_step probe heatmap, N = 64, seed 62](figure:heatmap/run-20261012-043022-3319/accel_step)

![force probe heatmap, N = 64, seed 44 (force never beats control)](figure:heatmap/run-20261011-234250-6e25/force)

**Where each quantity is read out:**

- **`accel_step`** usually peaks at the last layer: layer 4 in most 4-layer models, and layer 6 in the 6-layer model [f-20261012-030233-ab28, f-20261012-034134-5e82]. The exceptions are:
  - seed 67, which peaked at layer 2 [f-20261012-054929-0067];
  - the $N = 16$, 3000-step runs, which peaked at layer 2 [f-20261013-000329-a86b];
  - the 2-layer model, which peaked at layer 2, its last layer [f-20261013-000220-ab2d].
- **`distance`** peaks at layer 1 or 2 [f-20261011-223027-8fb3, f-20261012-002713-8d93].
- **Force** has no stable best layer. It ranges from layer 0 to layer 4 [f-20261011-201533-0653, f-20261012-002713-8d93] and drifts across layers 1–6 in the 6-layer model [f-20261012-030233-ab28].

### When the `accel_step` code forms

The `accel_step` code forms within a fairly narrow window of training steps. The loss at that point varies a lot between settings.

**Early runs.**

- The code was present at the first checkpoint (step 250), 1250 steps before `val_mse_ratio` fell below 0.1 [f-20261011-201533-0653].
- With 25-step checkpoints, it formed at step 100 in both seeds, while `val_mse_ratio` was still 1.96 and 1.84 [f-20261011-203659-4e2f].

**$N = 64$, $e \le 0.5$.** The persistent 0.1-bar crossing came at:

- steps 140, 140 and 100 [f-20261012-002713-8d93];
- steps 85–90 [f-20261012-005950-42a7];
- steps 90, 100 and 100, with half-final-margin steps of 80, 60 and 90 [f-20261012-044142-d490].

In each case, the crossing came 40–90 steps before the model beat constant-velocity extrapolation.

**Changing $N$ barely moves the formation step.**

- At $N = 16$, formation came at steps 90, 75 and 80, within 5–15 steps of the seed-matched $N = 64$ runs. But `val_mse_ratio` dropped below 1 at step 20, so the "forms before beating the baseline" ordering reverses [f-20261012-015955-fc2b].
- At $N = 256$ and $e \in [0, 0.85]$, formation came at steps 60, 70 and 60, even though the model never beat the baseline (final `val_mse_ratio` 2.03–2.37) [f-20261012-034134-5e82].
- The matched $N = 64$ arm formed at steps 60, 60 and 70 [f-20261012-034018-5526].

**Changing eccentricity.**

- At $e \in [0, 0.7]$, formation came at steps 60, 60 and 70 [f-20261012-055051-b178].
- At $e \in [0, 0.8]$, it came at steps 50–100 [f-20261013-000359-37e2].

**Comparing against the baseline is unreliable across $N$.** "Beats the baseline" depends on how strong the constant-velocity baseline is at each $N$, so it should not be used to time formation across $N$ [f-20261012-015955-fc2b].

**The loss level at formation is not fixed.** Formation came at:

- 2.0–2.9% of the step-0 value at $N = 64$, $e \le 0.5$ [f-20261012-044142-d490];
- 8–10% at $e \in [0, 0.85]$ [f-20261012-034018-5526];
- about 15–18% at $N = 256$ [f-20261012-034134-5e82].

Formation timing therefore tracks a window of training steps, not a fixed fraction of the loss reduction.

**A recurring reporting lag.** `formed_at_step` sometimes trails the trajectory by one checkpoint. In seed 68 it reports 80, while the margin holds from step 65 [f-20261012-054929-0067]. In the $N = 16$, 3000-step runs it reports 1000, while the trajectory first crosses 0.1 at step 500 [f-20261013-000329-a86b]. Where the two disagree, the findings use trajectory values.

![Formation timeline, N = 16, seed 47](figure:formation/run-20261012-015504-ae8c)

![Formation timeline, N = 256, e ∈ [0, 0.85], seed 59](figure:formation/run-20261012-033409-533d)

**Three timing hypotheses were refuted:**

- **Stricter bar at $N = 64$** [f-20261012-005950-42a7].
  - In 2 of 3 seeds, the `accel_step` margin never reached a 0.15 bar; it levelled off at 0.134–0.135.
  - The early force spike survived best-of-5 controls.
- **Halving the learning rate to 0.0005** [f-20261012-044258-ab88].
  - Formation moved from steps 60/60/70 to 70/80/100. That is +10/+20/+30 steps, not the roughly 2× delay that proportional scaling predicts.
  - Training did slow: final `val_mse_ratio` was about 0.22, against about 0.13 at lr 0.001.
- **Pre-registered $N = 64$ window** [f-20261012-054929-0067].
  - Only 2 of 5 fresh seeds (65–69) fell inside the window; at least 4 of 5 were required.
  - Seeds 67 and 68 formed for good at step 65, and seeds 65, 66 and 69 at step 90. Half-final steps ranged from 40 to 70.
  - These runs used 5-step checkpoints, against 10-step checkpoints earlier, so part of the shift to earlier steps may be resolution.

### Eccentricity sets the control level, not decoding quality

The layer-4 `accel_step` random-control $R^2$ falls as the eccentricity range widens:

- 0.71–0.77 at $e \le 0.5$;
- 0.543, 0.553 and 0.559 at $e \le 0.7$ [f-20261012-055051-b178];
- 0.372, 0.396 and 0.375 at $e \le 0.8$ [f-20261013-000359-37e2];
- 0.25–0.35 at $e \le 0.85$ [f-20261012-034018-5526].

Raw `accel_step` $R^2$ falls with it:

- 0.80–0.81 at $e \le 0.7$ [f-20261012-055051-b178];
- 0.670–0.672 at $e \le 0.8$ [f-20261013-000359-37e2];
- 0.51–0.59 at $e \le 0.85$ [f-20261012-034018-5526].

The larger raw margins at high $e$ therefore reflect more headroom, not a stronger representation.

The headroom-normalised margin also declines:

- 0.475, 0.455 and 0.475 at $e \le 0.8$. All three seeds fall in the pre-registered [0.44, 0.53] band, between the cited ~0.55–0.58 at $e \le 0.7$ and ~0.38–0.41 at $e \le 0.85$ [f-20261013-000359-37e2].
- 0.34–0.40 at $e \le 0.85$, by that finding's own measure [f-20261012-034018-5526].
- Force reaches only 0.15–0.21 at $e \le 0.8$ [f-20261013-000359-37e2] and about 0.16–0.22 at $e \le 0.7$ [f-20261012-055051-b178].

Three ceiling values cannot tell a gradual decline from one that steepens; per unit of $e$, the drop looks steeper above 0.8. The reference points also come from separate experiments that share one data seed [f-20261013-000359-37e2, f-20261012-055051-b178].

### Coarser sampling does not strengthen the `accel_step` code

**The prediction.** At $N = 16$ trained for 3000 steps, the `accel_step` margin would rise above the $N = 64$ reference [f-20261013-000329-a86b].

**The result: refuted in both seeds.**

- `accel_step` $R^2$ was 0.941 vs control 0.810 (margin 0.131) and 0.936 vs 0.795 (margin 0.140).
- The normalised margins of 0.69 and 0.68 fall below the predicted 0.8 and level with the ~0.70 at $N = 64$.
- Force led on raw margin: 0.236 and 0.275.
- Both models beat the baseline by a wide margin (`val_mse_ratio` 0.0007).

**Caveat.** The $N = 64$ reference runs used single controls and may differ in training length, so a matched arm is still needed [f-20261013-000329-a86b].

### Refuted hypotheses about the force probe

- **A wider mass range did not make force fall behind `accel_step` on raw margin.** The normalised margin still favoured `accel_step` (see above) [f-20261011-203628-5cd9].
- **Fixing $m_2 = 0.5$ did not lift force $R^2$ to the predicted 0.85** [f-20261011-223000-b5f9].
  - Force $R^2$ was 0.750 and 0.732, while `distance` was decoded at 0.87–0.89.
  - Both runs were cut short by the budget hard stop. Seed 14 reached 0.55 completion, below the 80% replication bar, so this finding effectively rests on seed 15 (completion 0.917).

### Linear decodability stops tracking prediction quality

The 6-layer, $d_{\text{model}} = 128$ model was trained at $e \in [0, 0.85]$ [f-20261012-030233-ab28]:

- After step 2000, layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 (seed 56) and from 0.683 to 0.705 (seed 57).
- Over the same steps, `val_mse_ratio` fell 6.8× and 5.5×, to 0.0037 and 0.0032.

The 2-layer model showed a related pattern: its `accel_step` margin plateaued by step 1500 while its force margin kept rising [f-20261013-000220-ab2d].

![accel_step probe heatmap, 6-layer d = 128 model, seed 56](figure:heatmap/run-20261012-023515-4712/accel_step)

### Out-of-distribution transfer

**A model trained on a restricted $e$ range did not stay within 10× of its in-distribution error on high-$e$ systems** [f-20261011-203723-b7de].

- On high-$e$ systems, `val_mse_ratio` was 0.270 and 0.249, about 16–18× the in-distribution value. It still stayed below 1.
- Transfer to low mass ratios held, at about 3.6× in-distribution.
- Three-body ratios of 72 and 70 are exploratory only.

**Training on $e \in [0, 0.8]$ closed the prediction gap** [f-20261011-211551-d718].

- The high-$e$ to in-distribution ratio was about 1.05× and 1.01×.
- Both runs were cut short by the budget hard stop at step 2500 (completion 0.867 and 0.9), with `val_mse_ratio` still falling.
- A probe gap remained: force $R^2$ was 0.366–0.377 on held-out high-$e$ systems, against 0.729–0.802 on low-$e$ systems. The out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $1/r^2$

- **At $e \in [0, 0.9]$:** the fits to the `accel_step` direction were $0.0103\,a^2/r$ and $0.00277\,((a+0.685)^3/r - r)$, with fit errors 0.71 and 0.62. These rest on weak probes ($R^2$ 0.395 and 0.503) [f-20261011-223027-8fb3].
- **With stronger probes ($R^2$ 0.831 and 0.881):** the fits were still $r^{-1}$, of the form $a^2/r - c\,a$ with $c \approx 0.55$ (fit errors 0.245 and 0.201). The force fits were $0.81/r$ and $0.80/r$ [f-20261011-231118-7a37].
- **Open check:** the fitting pipeline has not yet been run on the true `accel_step` target. Until it is, an artefact of the fitter cannot be ruled out [f-20261011-231118-7a37].

### Null and negative results, and open gaps

**Eight hypotheses were refuted:**

- a wider $m_2$ range would separate force from `accel_step` [f-20261011-203628-5cd9];
- restricted-$e$ training would transfer to high $e$ [f-20261011-203723-b7de];
- fixing $m_2$ would lift force [f-20261011-223000-b5f9];
- wide-$e$ training would yield $r^{-2}$ [f-20261011-223027-8fb3];
- at $N = 64$, `accel_step` would clear a 0.15 bar and the early force spike would not survive best-of-5 controls [f-20261012-005950-42a7];
- formation would scale with the learning rate [f-20261012-044258-ab88];
- $N = 64$ formation would fall in a narrow window [f-20261012-054929-0067];
- coarser sampling ($N = 16$) would raise the `accel_step` margin [f-20261013-000329-a86b].

**Not yet reported:**

- `velocity` probe results;
- activation-patching results;
- any test of whether the `accel_step` code is causally used.

**Limits throughout:**

- All probes are linear.
- Where `accel_step` controls are high ($e \le 0.5$, $N = 16$), its margins sit close to the 0.1 bar.
- Most results share data seed 0 and one 4-layer architecture.
