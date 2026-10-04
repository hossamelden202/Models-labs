from modellab.runtimes.models import RuntimeRequirement
from modellab.runtimes.registry import get_runtime, list_runtimes
from modellab.runtimes.resolver import resolve_runtime


def test_runtime_registry_contains_core_runtimes():

    runtime_ids = {
        runtime.runtime_id
        for runtime in list_runtimes()
    }

    assert "pytorch" in runtime_ids
    assert "ultralytics" in runtime_ids
    assert "transformers" in runtime_ids
    assert "timm" in runtime_ids
    assert "tensorflow" in runtime_ids
    assert "onnxruntime" in runtime_ids


def test_ultralytics_definition():

    runtime = get_runtime("ultralytics")

    assert runtime is not None
    assert runtime.package == "ultralytics"
    assert runtime.import_name == "ultralytics"


def test_unknown_runtime_is_not_automatically_accepted():

    result = resolve_runtime(
        RuntimeRequirement(
            runtime_id="definitely-not-a-real-runtime",
        )
    )

    assert result.state == "unknown"
    assert result.errors
