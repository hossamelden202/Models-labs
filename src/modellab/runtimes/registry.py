from __future__ import annotations

from .models import RuntimeDefinition


# ------------------------------------------------------------------
# IMPORTANT:
#
# These are CONTROLLED ModelLab runtimes.
#
# This registry is intentionally small initially. More runtimes can
# be added without changing the resolver/provisioner architecture.
# ------------------------------------------------------------------

_RUNTIMES: dict[str, RuntimeDefinition] = {

    "pytorch": RuntimeDefinition(
        runtime_id="pytorch",
        display_name="PyTorch",
        package="torch",
        import_name="torch",
        package_spec="torch",
        description="PyTorch model runtime.",
        capabilities=(
            "pytorch",
            "torchscript",
            "state_dict",
            "pickle_model",
        ),
        tags=("deep-learning", "vision", "gpu"),
    ),

    "ultralytics": RuntimeDefinition(
        runtime_id="ultralytics",
        display_name="Ultralytics",
        package="ultralytics",
        import_name="ultralytics",
        package_spec="ultralytics",
        description="Ultralytics YOLO model runtime.",
        capabilities=(
            "yolo",
            "object-detection",
            "segmentation",
            "classification",
            "pose",
        ),
        tags=("yolo", "computer-vision", "pytorch"),
    ),

    "transformers": RuntimeDefinition(
        runtime_id="transformers",
        display_name="Hugging Face Transformers",
        package="transformers",
        import_name="transformers",
        package_spec="transformers",
        description="Hugging Face Transformers runtime.",
        capabilities=(
            "transformers",
            "huggingface",
            "vision-transformer",
            "language-model",
        ),
        tags=("huggingface", "nlp", "vision"),
    ),

    "timm": RuntimeDefinition(
        runtime_id="timm",
        display_name="timm",
        package="timm",
        import_name="timm",
        package_spec="timm",
        description="PyTorch image models runtime.",
        capabilities=(
            "image-classification",
            "vision",
            "pytorch",
        ),
        tags=("computer-vision", "pytorch"),
    ),

    "tensorflow": RuntimeDefinition(
        runtime_id="tensorflow",
        display_name="TensorFlow",
        package="tensorflow",
        import_name="tensorflow",
        package_spec="tensorflow",
        description="TensorFlow runtime.",
        capabilities=(
            "tensorflow",
            "keras",
            "image-classification",
        ),
        tags=("deep-learning", "tensorflow"),
    ),

    "onnxruntime": RuntimeDefinition(
        runtime_id="onnxruntime",
        display_name="ONNX Runtime",
        package="onnxruntime",
        import_name="onnxruntime",
        package_spec="onnxruntime",
        description="ONNX Runtime CPU runtime.",
        capabilities=(
            "onnx",
            "inference",
        ),
        tags=("onnx", "inference"),
    ),
}


def list_runtimes() -> list[RuntimeDefinition]:
    return list(_RUNTIMES.values())


def get_runtime(runtime_id: str) -> RuntimeDefinition | None:
    return _RUNTIMES.get(runtime_id)
