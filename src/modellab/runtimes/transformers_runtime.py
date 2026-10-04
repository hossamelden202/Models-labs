from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class TransformersRuntimeError(RuntimeError):
    """Raised when the controlled Transformers runtime fails."""


class TransformersRuntime:
    """
    Controlled Hugging Face Transformers runtime.

    This class never imports Transformers into the ModelLab
    server process. All Transformers execution happens through
    the controlled runtime wrapper.
    """

    runtime_id = "transformers"
    provider_id = "huggingface"

    def __init__(
        self,
        environment: str | Path = (
            "/kaggle/working/modellab_runtime/transformers"
        ),
    ) -> None:

        self.environment = Path(environment)
        self.python = (
            self.environment
            / "bin"
            / "python"
        )

        if not self.python.exists():
            raise FileNotFoundError(
                f"controlled Transformers runtime not found: "
                f"{self.python}"
            )

    @property
    def manifest_path(self) -> Path:
        return self.environment / "modellab-runtime.json"

    def verify(self) -> dict[str, Any]:
        script = """
import json
import sys
import transformers

from transformers import AutoConfig, AutoModel

print(json.dumps({
    "python": sys.executable,
    "transformers": transformers.__version__,
    "AutoConfig": True,
    "AutoModel": True,
}))
"""

        result = self._run_script(script)

        return json.loads(
            result.stdout.strip()
        )

    def inspect_config(
        self,
        model_path: str | Path,
    ) -> dict[str, Any]:

        model_path = Path(model_path)

        script = """
import json
import sys

from transformers import AutoConfig

path = sys.argv[1]

config = AutoConfig.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=False,
)

data = config.to_dict()

print(json.dumps(data))
"""

        result = self._run_script(
            script,
            args=[str(model_path)],
        )

        return json.loads(
            result.stdout.strip()
        )

    def load_auto_model(
        self,
        model_path: str,
        *,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> dict[str, Any]:

        script = """
import json
import sys

from transformers import AutoConfig, AutoModel

path = sys.argv[1]
local_files_only = sys.argv[2] == "1"
trust_remote_code = sys.argv[3] == "1"

config = AutoConfig.from_pretrained(
    path,
    local_files_only=local_files_only,
    trust_remote_code=trust_remote_code,
)

model = AutoModel.from_pretrained(
    path,
    config=config,
    local_files_only=local_files_only,
    trust_remote_code=trust_remote_code,
)

print(json.dumps({
    "model_type": getattr(
        config,
        "model_type",
        None,
    ),
    "architectures": getattr(
        config,
        "architectures",
        None,
    ),
    "class_name": model.__class__.__name__,
}))
"""

        result = self._run_script(
            script,
            args=[
                str(model_path),
                "1" if local_files_only else "0",
                "1" if trust_remote_code else "0",
            ],
        )

        return json.loads(
            result.stdout.strip()
        )

    def _run_script(
        self,
        script: str,
        *,
        args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:

        args = list(args or [])

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="modellab_transformers_",
            delete=False,
            encoding="utf-8",
        ) as handle:

            handle.write(script)
            script_path = Path(handle.name)

        try:
            result = subprocess.run(
                [
                    str(self.python),
                    str(script_path),
                    *args,
                ],
                capture_output=True,
                text=True,
                check=False,
                env=self._environment(),
            )

            if result.returncode != 0:
                raise TransformersRuntimeError(
                    "controlled Transformers runtime failed\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            return result

        finally:
            script_path.unlink(
                missing_ok=True
            )

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()

        site_packages = (
            self.environment
            / "site-packages"
        )

        existing = env.get("PYTHONPATH")

        if existing:
            env["PYTHONPATH"] = (
                str(site_packages)
                + os.pathsep
                + existing
            )
        else:
            env["PYTHONPATH"] = str(
                site_packages
            )

        return env
