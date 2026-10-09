# Full Tools and Behavior of Model Lab

The complete reference for what Model Lab does and how each tool behaves. Each section links to a deeper document.

---

## 1. Purpose

Model Lab is a workbench for **image models**. It lets you register models and datasets, then evaluate, audit, and experiment, producing detailed reports on how models and datasets behave.

## 2. Model Registry ([details](MODEL_REGISTRY.md))

**Input methods**
- Upload the model file.
- Give a backend path (for example a Kaggle path when the backend runs on Kaggle).

**What happens:** Model Lab inspects the model, reconstructs it, and registers it.

**Metadata is mandatory:** a `.yml`, `.yaml`, or `.json` file with class names, number of classes, and input size, or manual entry in the UI.

**Support matrix**

| Model | Result |
|-------|--------|
| YOLO classification (any YOLO) | Works |
| YOLO detection (any YOLO) | Works |
| Transformer image models | Works, except models using `AutoModel` or depending on another model being present (not handled yet) |
| Any other type | Registration fails; upload a builder via Factory upload |

## 3. Dataset Registry ([details](DATASETS.md))

Register any dataset with `images/` and `labels/`.

## 4. Model Evaluation ([details](EVALUATION.md))

Runs a registered model on a registered dataset as a job. Parameters: `model_id`, `dataset_id`, `evaluation_id`, `batch_size`, `num_workers`, `device`, `seed`, `subpath`. Output includes `accuracy`, `macro_f1`, `weighted_f1`, `num_samples`.

## 5. Dataset Audit ([details](DATASET_AUDIT.md))

Needs only a dataset. Reports:
- class/object distribution and support per class
- precision, recall, and F1 per class
- exact duplicates
- dataset statistics and integrity
- what the dataset lacks

## 6. Experiments ([details](EXPERIMENTS.md))

Run experiments on a registered model; results are stored as jobs.

## 7. Jobs ([details](JOBS.md))

Evaluation, audit, and experiments run as jobs with `job_id`, `kind`, `params`, `status`, `result`, `error`, and timestamps.

## 8. End-to-end behavior

1. Register a model (upload or backend path + metadata).
2. Register a dataset (`images/` + `labels/`).
3. Audit the dataset.
4. Evaluate the model on the dataset.
5. Run experiments.
6. Read the stored job results.

## 9. Limitations

- Image models only.
- `AutoModel`-dependent or multi-model-dependent transformers are not handled yet.
- Unsupported model types fail registration; use a Factory-upload builder.
- Metadata (classes, class count, input size) must be provided.
