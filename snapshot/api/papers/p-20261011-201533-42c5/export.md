# Discovery Engine: running draft

## Abstract

When a small transformer predicts orbits well, does its residual stream hold the physical law or a shortcut that happens to work? And when in training does any real structure form? We train transformers on REBOUND two-body orbits ($G = M = 1$) sampled at $N$ snapshots per orbit. We then fit linear probes to the residual stream and judge each one against the best of several randomly initialised controls. The lab has 22 replicated findings (14 supported, 8 refuted) from 57 evidence runs; 4 of those runs were cut short by the budget hard stop. The models beat constant-velocity extrapolation, reaching `val_mse_ratio` as low as 0.0007 [f-20261013-000329-a86b]. The per-snapshot acceleration $(2\pi/N)^2 a^3/((1+m_2)r^2)$ beats its control in every finding. At $N = 64$ it forms at steps 65–140, often before the model beats the baseline [f-20261012-002713-8d93, f-20261012-054929-0067], and changing $N$ barely moves that step [f-20261012-015955-fc2b, f-20261012-034134-5e82]. Physical force $m_2/r^2$ does not beat its control in short runs (margins 0.000–0.080) [f-20261012-002713-8d93, f-20261012-044142-d490]. It emerges only after thousands of steps (margins 0.166–0.333) [f-20261011-203723-b7de, f-20261012-030233-ab28]. The main negative result: symbolic fits to the decoded directions give $r^{-1}$, not $r^{-2}$, in every seed tested [f-20261011-223027-8fb3, f-20261011-231118-7a37]. The fitter has not yet been checked on the true target.

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

The lab has 22 citable findings, all replicated across seeds: 14 supported and 8 refuted. They rest on 57 evidence runs. Four of those runs were cut short by the budget hard stop; each is flagged where its numbers are used. All probes below are linear. A probe "beats control" only if its $R^2$ exceeds the best random-init control by at least $\max(0.1, \text{control spread})$. The runs from the first day (2026-10-11) were judged against a single random control, not the best-of-several. Random controls disagree by less than 0.05 on `force` and `accel_step`, so this matters mainly for `distance` and `energy`.

### Prediction quality

Every model trained for thousands of steps beat constant-velocity extrapolation by a wide margin:

- about 0.011 for the default 4-layer model after 4000 steps [f-20261011-201533-0653];
- 0.0037 and 0.0032 for the 6-layer, $d_{\text{model}} = 128$ model [f-20261012-030233-ab28];
- 0.0007 at $N = 16$ after 3000 steps [f-20261013-000329-a86b].

Short runs were further from converged:

- 0.41–0.53 after 300 steps at $N = 64$ [f-20261012-002713-8d93, f-20261012-005950-42a7, f-20261012-044142-d490, f-20261012-054929-0067];
- about 0.13 after 600 steps at $e \in [0, 0.85]$ [f-20261012-034018-5526].

One arm never beat the baseline: at $N = 256$, $e \in [0, 0.85]$, the final `val_mse_ratio` was 2.03–2.37 [f-20261012-034134-5e82].

### The residual stream carries per-snapshot acceleration before it carries physical force

**`accel_step` beat its control in every finding.** It usually peaked at the last layer (layer 4 in 4-layer models, layer 6 in the 6-layer model). Typical scores were $R^2$ 0.88–0.95 at $e \le 0.5$, against controls of 0.71–0.81 [f-20261011-201533-0653, f-20261012-002713-8d93, f-20261012-044142-d490, f-20261013-000329-a86b].

**In short runs, physical force mostly did not beat its control.** Because $dt = P/N$, the step-based quantity $|a_{\text{step}}| = (2\pi/N)^2\,a^3/((1+m_2)\,r^2)$ is what next-step prediction needs; the physical force $|F| = m_2/r^2$ is not. Force margins were:

- 0.040, 0.000 and 0.062 at $N = 64$ [f-20261012-002713-8d93];
- 0.039–0.059 [f-20261012-005950-42a7];
- 0.033–0.080 [f-20261012-044142-d490];
- 0.065–0.078 at half learning rate [f-20261012-044258-ab88];
- 0.004–0.031 at $N = 256$ [f-20261012-034134-5e82].

There were a few borderline exceptions:

- 2 of 5 seeds at 0.103 and 0.105 [f-20261012-054929-0067];
- one of three $N = 16$ seeds at 0.123 [f-20261012-015955-fc2b];
- one of three $e \le 0.85$ seeds [f-20261012-034018-5526].

![accel_step probe heatmap, N = 64, seed 62](figure:heatmap/run-20261012-043022-3319/accel_step)

![force probe heatmap, N = 64, seed 44: force never beats its control](figure:heatmap/run-20261011-234250-6e25/force)

**Early force transient.** Force showed a brief spike near step 10 and then lost it, in every setting where it was tracked:

- margin 0.166–0.189 at step 10, decaying to about 0.00–0.02 by steps 40–50 [f-20261012-005950-42a7];
- 0.148–0.159 [f-20261012-044142-d490];
- 0.106–0.127 at $N = 16$ [f-20261012-015955-fc2b];
- 0.12–0.19 at steps 10–25 [f-20261012-054929-0067].

**With longer training, a force code does appear.** After 3000–4000 steps, force beat its control with margins of:

- 0.223 and 0.230 [f-20261011-201533-0653];
- 0.243 and 0.235 [f-20261011-203628-5cd9];
- 0.209 and 0.166 [f-20261011-203723-b7de];
- 0.221 and 0.255 in a 2-layer, $d = 32$ model [f-20261013-000220-ab2d];
- 0.236 and 0.275 at $N = 16$ [f-20261013-000329-a86b];
- 0.300 and 0.333 in the 6-layer model [f-20261012-030233-ab28].

In 600-step runs at higher eccentricity, force formed late (steps 250–400 at $e \le 0.8$; steps 300–500 at $e \le 0.75$) and was still rising at the end [f-20261013-000359-37e2, f-20261013-005923-526c]. In the 2-layer model, the force margin kept rising from step 500 to step 3000, while the `accel_step` margin levelled off at 0.13–0.15 by step 1500 [f-20261013-000220-ab2d].

**Which code leads depends on the metric.** On raw margin, force often exceeds `accel_step` in long runs. That is because the `accel_step` control is high, which caps its margin. Measured as a fraction of available headroom, $(R^2 - c)/(1 - c)$:

- `accel_step` leads 0.70 vs 0.48 and 0.71 vs 0.42 [f-20261011-203628-5cd9];
- `accel_step` leads 0.66/0.61 vs 0.41 [f-20261012-030233-ab28];
- the 2-layer model is split, one seed each way [f-20261013-000220-ab2d].

**Possible confound.** Force peaks at no stable layer: anywhere from layer 0 to layer 4, and across layers 1–6 in the 6-layer model [f-20261012-002713-8d93, f-20261012-030233-ab28]. It is also collinear with `accel_step` and `distance` through $1/r^2$, so some of the force margin may be a read of distance [f-20261011-203628-5cd9, f-20261013-000220-ab2d]. No run has yet regressed distance out of the force target.

### When the `accel_step` code forms

**`distance` and `energy` form first,** at steps 10–30, while the model is still far worse than the baseline (`val_mse_ratio` 5 to 83) [f-20261011-203659-4e2f, f-20261012-002713-8d93].

**`accel_step` follows within a narrow window of training steps.** The persistent 0.1-margin crossing came at:

- step 100 with 25-step checkpoints [f-20261011-203659-4e2f];
- steps 100–140 [f-20261012-002713-8d93];
- steps 85–90 [f-20261012-005950-42a7];
- steps 90–100 [f-20261012-044142-d490];
- steps 65–90 with 5-step checkpoints [f-20261012-054929-0067].

At $N = 64$, $e \le 0.5$, these crossings came 40–90 steps before the model first beat constant-velocity extrapolation.

**Changing $N$ barely moves the formation step, but it does move the baseline crossing:**

- At $N = 16$, formation came at steps 90, 75 and 80, within 5–15 steps of the seed-matched $N = 64$ runs. But `val_mse_ratio` fell below 1 at step 20, so the "forms before beating baseline" ordering reverses. "Beats baseline" is therefore not a usable formation marker across $N$ [f-20261012-015955-fc2b].
- At $N = 256$ and $e \in [0, 0.85]$, formation came at steps 60, 70 and 60, against 60, 60 and 70 for the matched $N = 64$ arm, even though the $N = 256$ model never beat the baseline [f-20261012-034134-5e82, f-20261012-034018-5526].

**The loss level at formation is not fixed.** It was 2.0–2.9% of the step-0 value at $N = 64$, $e \le 0.5$ [f-20261012-044142-d490], 8–10% at $e \le 0.85$ [f-20261012-034018-5526] and about 15–18% at $N = 256$ [f-20261012-034134-5e82]. Formation timing tracks a window of training steps, not a fixed fraction of the loss reduction.

**The learning rate shifts formation only slightly.** Halving it delayed formation by +10/+20/+30 steps, not the roughly 2× that proportional scaling predicts [f-20261012-044258-ab88].

![Formation timeline, N = 16, seed 47](figure:formation/run-20261012-015504-ae8c)

![Formation timeline, N = 256, e ∈ [0, 0.85], seed 59](figure:formation/run-20261012-033409-533d)

### Eccentricity sets the control level

As the eccentricity range widens, both the `accel_step` control $R^2$ and the trained $R^2$ fall. The margins therefore grow because there is more headroom, not because decoding improves.

| $e$ range | control $R^2$ | trained $R^2$ | normalised margin |
|---|---|---|---|
| $\le 0.5$ | 0.71–0.77 | 0.88–0.95 | — |
| $\le 0.7$ | 0.543–0.559 | 0.80–0.81 | ≈0.55 |
| $\le 0.75$ | 0.463–0.487 | 0.746–0.753 | 0.51–0.54 |
| $\le 0.8$ | 0.372–0.396 | 0.670–0.672 | 0.455–0.475 |
| $\le 0.85$ | 0.25–0.35 | 0.51–0.59 | 0.34–0.40 |

Sources: [f-20261012-055051-b178, f-20261013-005923-526c, f-20261013-000359-37e2, f-20261012-034018-5526]. Each pre-registered band was met.

The decline does not look uniform. About 0.05 of $R^2$ is lost from $e \le 0.7$ to $e \le 0.75$, then about 0.19 from $e \le 0.75$ to $e \le 0.85$, which suggests it steepens above 0.75 [f-20261013-005923-526c]. All arms share data seed 0.

### Linear decodability stops tracking prediction quality

In the 6-layer model at $e \in [0, 0.85]$ [f-20261012-030233-ab28]:

- after step 2000, layer-6 `accel_step` $R^2$ rose only from 0.710 to 0.735 and from 0.683 to 0.705, never reaching 0.8;
- over the same steps, `val_mse_ratio` fell 6.8× and 5.5×.

![accel_step heatmap, 6-layer d = 128 model, seed 56](figure:heatmap/run-20261012-023515-4712/accel_step)

### Out-of-distribution transfer

**Trained on $e \le 0.5$:** high-$e$ error was 16–18× the in-distribution error (`val_mse_ratio` 0.270 and 0.249). That misses the pre-registered 10× criterion, although it stays below 1. Transfer to low mass ratios held at about 3.6× in-distribution [f-20261011-203723-b7de].

**Trained on $e \in [0, 0.8]$:** the gap closed, at about 1.05× and 1.01× [f-20261011-211551-d718].

- Both runs were cut short by the budget hard stop at step 2500 (completion 0.867 and 0.9), with `val_mse_ratio` still falling.
- A probe gap remained: force $R^2$ was 0.366–0.377 on held-out high-$e$ systems, against 0.729–0.802 on low-$e$ systems. The out-of-distribution probes have no random controls.

### Symbolic regression has not recovered $r^{-2}$

- **With strong probes** ($R^2$ 0.831 and 0.881), the fits to the `accel_step` direction were $r^{-1}$, of the form $a^2/r - c\,a$ with $c \approx 0.55$ (fit errors 0.245 and 0.201). The force fits were $0.81/r$ and $0.80/r$ [f-20261011-231118-7a37].
- **At $e \le 0.9$, with weak probes** ($R^2$ 0.395 and 0.503), the fits were also $r^{-1}$ [f-20261011-223027-8fb3].
- **Open check:** the fitter has not yet been run on the true `accel_step` target, so a fitting artefact cannot be ruled out [f-20261011-231118-7a37].

### Null and negative results

Eight hypotheses were refuted:

1. **A wider $m_2$ range would make force fall behind `accel_step`.** Force kept the larger raw margin [f-20261011-203628-5cd9].
2. **A model trained on $e \le 0.5$ would transfer to high $e$ within 10×** [f-20261011-203723-b7de].
3. **Fixing $m_2 = 0.5$ would lift force $R^2$ to 0.85.** It reached 0.750 and 0.732, while `distance` was decoded at 0.87–0.89 [f-20261011-223000-b5f9].
   - Both runs were cut short by the budget hard stop. Seed 14 reached only 0.55 completion, below the 80% replication bar, so this result effectively rests on seed 15 (completion 0.917).
4. **Wide-$e$ training would yield an $r^{-2}$ symbolic fit** [f-20261011-223027-8fb3].
5. **At $N = 64$, `accel_step` would clear a 0.15 bar and the force spike would vanish under best-of-5 controls.** Neither held [f-20261012-005950-42a7].
6. **Formation would scale in proportion to the learning rate** [f-20261012-044258-ab88].
7. **$N = 64$ formation would fall in a narrow pre-registered window.** Only 2 of 5 fresh seeds fell inside; at least 4 were required. The shift may partly reflect finer checkpoint spacing [f-20261012-054929-0067].
8. **Coarser sampling ($N = 16$) would raise the `accel_step` margin.** The normalised margin was 0.69 and 0.68, level with $N = 64$ [f-20261013-000329-a86b].

### Not yet reported, and limits

**Not yet reported:**

- `velocity` probe results;
- activation patching;
- any causal test of whether the `accel_step` code is used.

**Limits throughout:**

- All probes are linear.
- Where `accel_step` controls are high ($e \le 0.5$, $N = 16$), its margins sit close to the 0.1 bar.
- `formed_at_step` sometimes trails the trajectory by one checkpoint; the findings use trajectory values where the two disagree [f-20261012-054929-0067, f-20261013-000329-a86b].
- Most results share data seed 0 and a single 4-layer architecture.
