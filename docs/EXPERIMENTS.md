# Experiments

Experiments let you run structured runs against a registered model, beyond a single evaluation. Each experiment runs as a [job](JOBS.md) and its result is stored with it.

---

## Job shape

Experiment jobs use the same job envelope as evaluations:

```json
{
  "job_id": "<id>",
  "kind": "<experiment kind>",
  "status": "completed",
  "params": { "model_id": "...", "dataset_id": "...", "seed": 42 },
  "result": { },
  "error": null,
  "created_at": "...",
  "started_at": "...",
  "finished_at": "..."
}
```

> TODO: the experiment sample provided so far was identical to the evaluation job (`kind: "evaluation"`). Replace the placeholders above with a real experiment job (its `kind`, `params`, and `result`) before publishing.

---

## Common params

| Param | Meaning |
|-------|---------|
| `model_id` | Registered model under test |
| `dataset_id` | Registered dataset used |
| `device` | `auto` or an explicit device |
| `batch_size` | Batch size |
| `num_workers` | Data loading workers |
| `seed` | Seed for reproducibility |

## Relationship to other tools

- [Evaluation](EVALUATION.md) answers: *how does this model score on this dataset?*
- Experiments answer: *how does this model behave when I change something?*
- [Dataset Audit](DATASET_AUDIT.md) answers: *is this dataset any good?*
