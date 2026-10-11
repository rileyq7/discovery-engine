# Task: propose the next experiments

Below is the lab's current state (findings, open questions, recent experiments with their results,
budget) and the JSON schema experiment configs must satisfy.

Propose 1 to {max_proposals} experiments that most efficiently advance the research question given what
is already known. Avoid repeating experiments that already answered their question unless you are
replicating with new seeds. Fit within the budget: the remaining budget today is {budget_left} compute-hours,
the auto-approval cap is {cap} hours per experiment (anything larger waits for a human), and the compute
estimate is roughly proportional to steps x batch_size x context_len x n_layers x d_model^2.
Use `"probes": {{"targets": [...]}}` with `accel_step` alongside `force` whenever you probe gravity.

Families: E1 baseline prediction, E2 force probe, E3 law extraction (probes.symbolic = true),
E4 transfer (eval_sets with unseen mass ratios / n_bodies 3), E5 formation timing (dense checkpoints),
E6 scaling sweep cell, custom for anything else.

A reference config that is known to work (E2, ~0.45 compute-hours for 2 seeds on this machine):
```json
{reference_config}
```

Respond with ONLY a JSON object, no prose:
```json
{{"reasoning": "<2-4 sentences: why these experiments now>",
  "proposals": [<experiment config>, ...],
  "new_questions": ["<optional new open question>", ...]}}
```

## Lab state
```json
{context}
```

## Config JSON schema
```json
{schema}
```
