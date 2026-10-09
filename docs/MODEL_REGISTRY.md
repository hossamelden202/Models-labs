# Model Registry

The Model Registry is where models enter Model Lab. Registration means Model Lab **inspects** the model, **reconstructs** it so it can run, and stores it under a `model_id` that other tools (evaluation, experiments) refer to.

> Scope: **image models only.**

---

## 1. Ways to register a model

### A. Upload
Upload the model file through the app. Model Lab inspects and reconstructs it.

### B. Backend path
Instead of uploading, give a path that the **backend** can see.

Example: if the backend runs on Kaggle, provide the model's **Kaggle path**. Nothing needs to be uploaded; the backend reads it in place.

Use this when the model is large, or when it already lives where the backend runs.

---

## 2. Required metadata

Every model must come with information about itself. Provide it in one of two ways.

### Option 1: a metadata file (`.yml`, `.yaml`, or `.json`)

The file must describe at least:

| Field | Meaning |
|-------|---------|
| class names | Names of the classes the model predicts |
| number of classes | How many classes there are |
| input size | The image size the model expects |

Illustrative example (adapt the key names to your Model Lab version):

```yaml
names:
  0: cat
  1: dog
  2: bird
num_classes: 3
input_size: 224
```

```json
{
  "names": ["cat", "dog", "bird"],
  "num_classes": 3,
  "input_size": 224
}
```

### Option 2: manual registration in the UI
Skip the file and enter the class names, number of classes, and input size by hand in the UI.

---

## 3. What is supported

| Model type | Registration |
|------------|--------------|
| YOLO classification (any YOLO model) | Confirmed working |
| YOLO detection (any YOLO model) | Confirmed working |
| Transformer image models | Confirmed working, with the exception below |

### Transformer exception
A transformer model registers successfully **unless** it uses `AutoModel`, or otherwise refers to **another model that must be present**. That case is **not handled yet**.

---

## 4. What is not supported (and the workaround)

- Models that depend on `AutoModel` or on another model being present: not handled yet.
- Any other model type: **registration fails**.

**Workaround: Factory upload.** For a model that fails to register, the user can upload a **builder** for that model in the Factory upload. The builder tells Model Lab how to construct the model, so it can be registered even though automatic reconstruction cannot handle it.

---

## 5. Registration behavior summary

1. You choose upload or backend path.
2. You provide metadata (file or UI).
3. Model Lab inspects the model and reconstructs it.
4. On success the model is registered and gets a `model_id`.
5. On failure, use the Factory upload with a builder.

Registered models are then available to [Evaluation](EVALUATION.md) and [Experiments](EXPERIMENTS.md).
