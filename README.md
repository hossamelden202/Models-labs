# ModelLab

## Failure analysis

`modellab analyze --evaluation <id> [--audit <id>]` reads a stored evaluation (and optionally an
audit) and writes `failure_analyses/<id>/`: a per-sample failure table, a slice table with
Benjamini-Hochberg corrected evidence grades, confusion analysis, optional failure clusters, a
deterministic `report.json`, and a human-readable `summary.txt`.

Slices represent associations and evidence, not proven causes.

## Controlled experiments

`modellab.experiments` runs controlled interventions against an immutable, fingerprinted baseline
using `create_baseline`, `run_experiment`, `run_family`, and `run_matrix`.

Supported intervention types include:

- Image transforms
- Preprocessing changes
- Inference interventions such as temperature scaling, confidence/class thresholds, and class weights
- Optional restriction to a failure slice

Data interventions that require retraining are rejected because ModelLab currently has no training
interface.

Experiments use paired comparisons including McNemar's test, bootstrap confidence intervals, and
sign-flip permutation tests. Family-level results support multiple-comparison correction.

Results are stored under:

`experiments/<family>/<experiment_id>/`

The experiment system reports controlled effects and associations; it does not automatically infer
root causes, generate hypotheses, retrain models, or perform automated repairs.
