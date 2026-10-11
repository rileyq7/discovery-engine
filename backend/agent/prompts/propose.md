# Task: propose the next experiments

Below is the lab's current state (findings, open questions, recent experiments with their results,
budget) and the JSON schema experiment configs must satisfy.

Propose 1 to {max_proposals} experiments that most efficiently advance the research question given what
is already known. Avoid repeating experiments that already answered their question unless you are
replicating with new seeds. Fit within the budget: the remaining budget today is {budget_left} compute-hours,
the auto-approval cap is {cap} hours per experiment (anything larger waits for a human). On this laptop, training
time is dominated by about 0.03 s per step per layer (steps x n_layers), with a smaller term growing with
batch_size x context_len x d_model^2. Symbolic regression adds about 0.06 h per seed.
Use `"probes": {{"targets": [...]}}` with `accel_step` alongside `force` whenever you probe gravity.

Families: E1 baseline prediction, E2 force probe, E3 law extraction (probes.symbolic = true),
E4 transfer (eval_sets with unseen mass ratios / n_bodies 3), E5 formation timing (dense checkpoints),
E6 scaling sweep cell, custom for anything else.

A reference config that is known to work (E2, ~0.45 compute-hours for 2 seeds on this machine):
```json
{reference_config}
```

An independent reviewer checks your findings against the raw data every hour. If the lab state has
`open_review_issues`, you must answer every one in `review_responses`. Say what you concluded and what you are
doing about it: for example, propose the experiment that resolves it, or give the correct number. You can't
edit past findings. A response that names a `finding_id` is attached to that finding as a note. If you disagree
with an issue, say why. If you accept that a finding is overclaimed, also set `"set_status"` to its correct status
(`supported`, `refuted` or `inconclusive`). You can only correct downward; replication still has to be earned.

Respond with ONLY a JSON object, no prose:
```json
{{"reasoning": "<2-4 sentences: why these experiments now>",
  "proposals": [<experiment config>, ...],
  "new_questions": ["<optional new open question>", ...],
  "review_responses": [{{"review_id": "r-...", "issue": 0, "response": "...", "finding_id": "<optional f-...>", "set_status": "<optional>"}}]}}
```

## Lab state
```json
{context}
```

## Config JSON schema
```json
{schema}
```
