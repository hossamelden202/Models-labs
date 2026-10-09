# Model Lab

**Copyright (c) 2026 Hossam. All rights reserved.**

This repository is provided for viewing and evaluation purposes only. No permission is granted to copy, modify, distribute, sublicense, or use this software, in whole or in part, for commercial or other purposes without prior written permission from the copyright holder.

Access to this public repository does not grant any license or other rights to the software, except as required by applicable law.

**Register image models. Register datasets. Evaluate, audit, and experiment, all from one place.**

Model Lab is a workbench for image models (YOLO classification, YOLO detection, and transformer image models). You bring a model and a dataset; Model Lab inspects them, reconstructs the model, runs jobs against it, and gives you detailed reports on how the model and the data actually behave.

> Full reference: **[Full Tools and Behavior of Model Lab](docs/FULL_TOOLS_AND_BEHAVIOR.md)**

---

## The five tools

| # | Tool | What it does | Docs |
|---|------|--------------|------|
| 1 | **Model Registry** | Register a model by upload or by backend path. Model Lab inspects and reconstructs it. | [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md) |
| 2 | **Dataset Registry** | Register any dataset laid out as `images/` + `labels/`. | [docs/DATASETS.md](docs/DATASETS.md) |
| 3 | **Model Evaluation** | Run a registered model on a registered dataset and get detailed metrics. | [docs/EVALUATION.md](docs/EVALUATION.md) |
| 4 | **Dataset Audit** | Needs only a dataset. Reports integrity, duplicates, class distribution, and what the dataset lacks. | [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md) |
| 5 | **Experiments** | Run experiments on a model. | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |

All long-running work (evaluation, audit, experiments) runs as a **job** with a `job_id`, a status, and a stored result. See [docs/JOBS.md](docs/JOBS.md).

---

## Quick overview

### 1. Model Registry

- Two ways in: **upload** the model file, or provide a **backend path** (for example, if the backend runs on Kaggle, provide the model's Kaggle path).
- Model Lab **inspects and reconstructs** the model, then registers it.
- **Models must be image models.**
- Every model needs metadata: a `.yml`, `.yaml`, or `.json` file with class names, number of classes, and input size, **or** you can enter the same information manually in the UI.

**Confirmed working**

| Model type | Status |
|------------|--------|
| YOLO classification (any YOLO) | Working |
| YOLO detection (any YOLO) | Working |
| Transformer image models | Working, unless the model relies on `AutoModel` or refers to another model that must be present (see below) |

**Not handled yet**

- Models that use `AutoModel`, or that reference another model which must also be present.
- Any other model type fails to register automatically. Workaround: **upload a builder for that model in the Factory upload**.

### 2. Dataset Registry

Register any dataset that contains `images/` and `labels/`.

### 3. Model Evaluation

Pick a model and a dataset, run the evaluation, and get a metrics report (accuracy, macro F1, weighted F1, sample count, and more). Details in [docs/EVALUATION.md](docs/EVALUATION.md).

### 4. Dataset Audit

Give it a dataset and nothing else. You get a full report on integrity, exact duplicates, class distribution, support per class, per-class precision / recall / F1, and general dataset statistics. Details in [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md).

### 5. Experiments

Run experiments on a model and track them as jobs. Details in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## Typical workflow

```text
Register model  ─┐
                 ├─► Evaluate   ─► Metrics report
Register dataset ┤
                 ├─► Audit      ─► Dataset health report
                 └─► Experiment ─► Experiment results
```

1. Register a model (upload or backend path, plus metadata).
2. Register a dataset (`images/` + `labels/`).
3. Audit the dataset first, so you know what you are evaluating against.
4. Evaluate the model on the dataset.
5. Run experiments to dig deeper.

---

## Documentation index

- [Full Tools and Behavior of Model Lab](docs/FULL_TOOLS_AND_BEHAVIOR.md): The complete reference
- [Model Registry](docs/MODEL_REGISTRY.md)
- [Datasets](docs/DATASETS.md)
- [Evaluation](docs/EVALUATION.md)
- [Dataset Audit](docs/DATASET_AUDIT.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Jobs](docs/JOBS.md)

---

## Known limitations

- Image models only.
- Models depending on `AutoModel` or on another model being present are not handled yet.
- Unsupported model types fail registration; use a Factory-upload builder.
- Metadata (classes, class count, input size) must be supplied via file or UI.

---

## Copyright and Permissions

Copyright © 2026 Hossam. All rights reserved.

You may view this repository for informational and evaluation purposes. You may not copy, modify, redistribute, sublicense, or commercially exploit this software without prior written permission from the copyright holder, except where applicable law provides otherwise.

For permission requests, contact the copyright holder through the contact information provided on the GitHub profile.
