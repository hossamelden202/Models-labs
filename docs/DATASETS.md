# Datasets

Users can register any dataset that contains images and labels. A registered dataset gets a `dataset_id` that is used by [Evaluation](EVALUATION.md), [Dataset Audit](DATASET_AUDIT.md), and [Experiments](EXPERIMENTS.md).

---

## Required layout

```
my_dataset/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
└── labels/
    ├── img_0001.txt
    ├── img_0002.txt
    └── ...
```

- `images/` holds the image files.
- `labels/` holds the labels for those images.

Any dataset with this `images/` + `labels/` structure can be registered.

---

## Using a dataset

Once registered, the dataset can be:

| Use | Tool |
|-----|------|
| Run a model against it and measure behavior | [Evaluation](EVALUATION.md) |
| Inspect its quality and contents (no model needed) | [Dataset Audit](DATASET_AUDIT.md) |
| Base for experiments on a model | [Experiments](EXPERIMENTS.md) |

## Recommended order
Audit a dataset **before** evaluating on it. The audit surfaces duplicates, integrity problems, and class imbalance that directly affect how much you can trust evaluation numbers.
