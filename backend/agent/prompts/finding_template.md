# Task: analyze a finished experiment and record a finding

The experiment below has finished running and its probes are done. Read the hypothesis, the expected
results if true/false, and the measured results, then write ONE finding.

Guidance:
- verdict is `supported` if the expected-if-true outcome happened, `refuted` if the expected-if-false
  outcome happened, `inconclusive` otherwise (mixed seeds, borderline margins, runs stopped early,
  failed runs, or results the hypothesis did not anticipate).
- The claim is one or two sentences with the key numbers (e.g. "Force probe R2 = 0.31 at layer 3 vs
  control 0.28; accel_step probe R2 = 0.93 vs control 0.40"). Compare force vs accel_step whenever both exist.
- evidence_run_ids: the completed run IDs from this experiment that support the verdict (all completed
  runs, unless one is marked bad). Use only IDs listed below.
- confidence_note: what limits confidence and what would change the verdict.
- seeds_agree: true only if two or more completed runs with different seeds independently show the verdict.
- replicates: set to "{replicates}" if this experiment is a replication (otherwise null).
- question_updates: open questions this finding answers or bears on, e.g. {{"id": "q-...", "status": "answered"}}
  (use "open" to just link it).
- new_questions: follow-up questions this result raises (0-2).

Respond with ONLY a JSON object:
```json
{{"claim": "...", "verdict": "supported|refuted|inconclusive", "evidence_run_ids": ["run-..."],
  "confidence_note": "...", "seeds_agree": false, "replicates": null, "experiment_id": "{experiment_id}",
  "question_updates": [], "new_questions": []}}
```

## Open questions
```json
{questions}
```

## Experiment and results
```json
{results}
```
