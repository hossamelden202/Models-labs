# Jobs

Evaluations, dataset audits, and experiments all run as **jobs**. A job is a stored record with an id, parameters, a status, and a result.

## Job record

| Field | Meaning |
|-------|---------|
| `job_id` | Unique id, e.g. `5da64aa617a0` |
| `kind` | Type of job, e.g. `evaluation` |
| `params` | Everything the job was started with |
| `status` | Current state, e.g. `completed` |
| `result` | Output of the job (empty/`null` until finished) |
| `error` | `null` on success, otherwise the failure reason |
| `created_at` | When the job was submitted |
| `started_at` | When it began running |
| `finished_at` | When it ended |

All timestamps are ISO 8601 in UTC.

## Example

```json
{
  "created_at": "2026-10-07T13:54:50.409448+00:00",
  "error": null,
  "finished_at": "2026-10-07T13:56:11.908146+00:00",
  "job_id": "5da64aa617a0",
  "kind": "evaluation",
  "params": { "model_id": "demo_model2343", "dataset_id": "trans", "batch_size": 32, "device": "auto", "num_workers": 5, "seed": 42, "evaluation_id": "wer", "subpath": null },
  "result": { "accuracy": 0.7656, "macro_f1": 0.7614942870854636, "weighted_f1": 0.7614942870854635, "num_samples": 5000, "device": "auto", "evaluation_id": "wer" },
  "started_at": "2026-10-07T13:54:50.410524+00:00",
  "status": "completed"
}
```

## Reading a job
- `status: completed` and `error: null` means the `result` is valid.
- Duration = `finished_at` - `started_at` (about 81 seconds in the example).
- Re-running with the same `params` and `seed` should give comparable results.
