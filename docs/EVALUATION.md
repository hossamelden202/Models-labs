# Model Evaluation

Evaluation runs a **registered model** on a **registered dataset** and produces metrics that show how the model behaves.

Evaluation is a [job](JOBS.md): you submit it, it runs, and the result is stored with the job.

---

## Inputs (job `params`)

| Param | Meaning | Example |
|-------|---------|---------|
| `model_id` | The registered model to test | `"demo_model2343"` |
| `dataset_id` | The registered dataset to test on | `"trans"` |
| `evaluation_id` | Identifier of this evaluation run | `"wer"` |
| `batch_size` | Images per batch | `32` |
| `num_workers` | Data loading workers | `5` |
| `device` | Where to run; `auto` picks automatically | `"auto"` |
| `seed` | Random seed for reproducibility | `42` |
| `subpath` | Optional subpath inside the dataset | `null` |

---

## Real example

A completed evaluation job:

```json
{
  "created_at": "2026-10-07T13:54:50.409448+00:00",
  "error": null,
  "finished_at": "2026-10-07T13:56:11.908146+00:00",
  "job_id": "5da64aa617a0",
  "kind": "evaluation",
  "params": {
    "batch_size": 32,
    "dataset_id": "trans",
    "device": "auto",
    "evaluation_id": "wer",
    "model_id": "demo_model2343",
    "num_workers": 5,
    "seed": 42,
    "subpath": null
  },
  "result": {
    "accuracy": 0.7656,
    "device": "auto",
    "evaluation_id": "wer",
    "macro_f1": 0.7614942870854636,
    "num_samples": 5000,
    "weighted_f1": 0.7614942870854635
  },
  "started_at": "2026-10-07T13:54:50.410524+00:00",
  "status": "completed"
}
```

---

## Reading the result

| Field | Meaning |
|-------|---------|
| `accuracy` | Fraction of samples predicted correctly (here 76.56%) |
| `macro_f1` | F1 averaged equally across classes; sensitive to weak classes |
| `weighted_f1` | F1 averaged by class support; reflects overall dataset mix |
| `num_samples` | How many samples were evaluated (here 5000) |
| `device` | Device setting used |
| `evaluation_id` | Which evaluation this result belongs to |

**Tip:** when `macro_f1` is noticeably lower than `weighted_f1` (or accuracy), some rare classes are performing worse than the frequent ones. Cross-check with the [Dataset Audit](DATASET_AUDIT.md) class distribution.

---

## Goal of evaluation

Evaluation is meant to give **intensive metrics** so you can accurately learn how a model behaves on a given dataset, not just a single headline number. The fields above are the metrics shown in the current output; additional metrics appear in the `result` object as the evaluator produces them.

---

## Behavior notes

- Works with any registered model type (YOLO classification, YOLO detection, transformer image models).
- Results are stored with the job, including `started_at`, `finished_at`, and `error` (`null` on success).
- Use a fixed `seed` to make runs comparable.
