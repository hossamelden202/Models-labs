# Dataset Audit

The Dataset Audit needs **only a dataset**. No model is required. It inspects the dataset and returns a full report of everything that can be learned from it.

Like evaluation, it runs as a [job](JOBS.md).

---

## What the audit tells you

### Class and object distribution
- The distribution of the different types (classes) of objects in the dataset.
- The number of **support samples** for each class.

### Per-class quality
- **Precision**, **recall**, and **F1 score** for each class.

### Findings
The audit also reports findings about the dataset itself:

- **Distribution of objects**: how balanced or imbalanced the classes are.
- **Exact duplicates**: identical items that inflate the dataset and can leak between splits.
- **Dataset statistics**: general statistics describing the dataset.
- **Integrity**: whether the dataset is consistent and well-formed.
- **What the dataset lacks**: gaps and weaknesses worth fixing.

---

## Why run it

| Problem the audit can reveal | Why it matters |
|------------------------------|----------------|
| Class imbalance | Rare classes get low macro F1 in evaluation |
| Exact duplicates | Inflated sample counts, optimistic metrics |
| Integrity issues | Broken or mismatched images/labels cause silent errors |
| Missing coverage | Tells you what to collect next |

## Recommended use
Run the audit **before** evaluating a model on a dataset, and again after you modify the dataset, to confirm the problems are gone.

## Inputs
- `dataset_id` of a registered dataset (see [Datasets](DATASETS.md)).
