import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

numpy_stub = types.ModuleType("numpy")
numpy_stub.generic = type("generic", (), {})
numpy_stub.isfinite = lambda value: True
sys.modules.setdefault("numpy", numpy_stub)

pydantic_stub = types.ModuleType("pydantic")

class BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, *args, **kwargs):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = lambda *args, **kwargs: None
sys.modules.setdefault("pydantic", pydantic_stub)

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.nn = types.SimpleNamespace(Module=type("Module", (), {}))
torch_stub.load = lambda *args, **kwargs: {}
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)

from modellab.loading.artifact import extract_hints


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.sidebar = DummyContext()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


streamlit_stub = types.ModuleType("streamlit")
streamlit_stub.set_page_config = lambda *args, **kwargs: None
streamlit_stub.session_state = SessionState()
streamlit_stub.sidebar = DummyContext()
streamlit_stub.title = lambda *args, **kwargs: None
streamlit_stub.caption = lambda *args, **kwargs: None
streamlit_stub.divider = lambda *args, **kwargs: None
streamlit_stub.markdown = lambda *args, **kwargs: None
streamlit_stub.info = lambda *args, **kwargs: None
streamlit_stub.warning = lambda *args, **kwargs: None
streamlit_stub.success = lambda *args, **kwargs: None
streamlit_stub.error = lambda *args, **kwargs: None
streamlit_stub.json = lambda *args, **kwargs: None
streamlit_stub.expander = lambda *args, **kwargs: DummyContext()
streamlit_stub.metric = lambda *args, **kwargs: None
streamlit_stub.columns = lambda *args, **kwargs: []
streamlit_stub.radio = lambda *args, **kwargs: None
streamlit_stub.text_input = lambda *args, **kwargs: ""
streamlit_stub.button = lambda *args, **kwargs: False
streamlit_stub.form = lambda *args, **kwargs: DummyContext()
streamlit_stub.file_uploader = lambda *args, **kwargs: None
streamlit_stub.number_input = lambda *args, **kwargs: 0
streamlit_stub.selectbox = lambda *args, **kwargs: None
streamlit_stub.checkbox = lambda *args, **kwargs: False
streamlit_stub.spinner = lambda *args, **kwargs: DummyContext()
streamlit_stub.subheader = lambda *args, **kwargs: None
streamlit_stub.write = lambda *args, **kwargs: None
streamlit_stub.stop = lambda *args, **kwargs: None
streamlit_stub.form_submit_button = lambda *args, **kwargs: False
streamlit_stub.columns = lambda count: tuple(DummyContext() for _ in range(count))
streamlit_stub.session_state = SessionState()

requests_stub = types.ModuleType("requests")
requests_stub.RequestException = Exception
requests_stub.request = lambda *args, **kwargs: None

sys.modules.setdefault("streamlit", streamlit_stub)
sys.modules.setdefault("requests", requests_stub)

spec = importlib.util.spec_from_file_location("modellab_ui", ROOT / "modellab_ui.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestModelLabUI(unittest.TestCase):
    def test_artifact_resolution_defaults_follow_backend_inspection(self):
        payload = module.artifact_resolution_defaults(
            artifact_id="ck",
            inspection={
                "format": "torch_archive",
                "state_dicts": [{"key_path": "model_state_dict"}],
                "hints": [
                    {"field": "class_names", "value": ["cat", "dog"]},
                    {"field": "input_size", "value": [224, 224]},
                    {"field": "architecture_hint", "value": "vit"},
                ],
            },
        )

        self.assertEqual(payload["artifact_id"], "ck")
        self.assertEqual(payload["class_names"], ["cat", "dog"])
        self.assertEqual(payload["input_size"], [224, 224])
        self.assertEqual(payload["architecture"], "vit")
        self.assertEqual(payload["state_dict_key"], "model_state_dict")
        self.assertNotIn("source", payload)

    def test_self_contained_artifacts_do_not_force_state_dict_fields(self):
        payload = module.artifact_resolution_defaults(
            artifact_id="ts",
            inspection={
                "format": "torchscript",
                "state_dicts": [],
                "hints": [
                    {"field": "num_classes", "value": 3},
                    {"field": "input_size", "value": [224, 224]},
                ],
            },
        )

        self.assertEqual(payload["num_classes"], 3)
        self.assertEqual(payload["input_size"], [224, 224])
        self.assertNotIn("state_dict_key", payload)
        self.assertNotIn("architecture", payload)

    def test_shape_hint_infers_num_classes_but_not_class_names(self):
        hints = extract_hints({"output_shape": [1, 1001]})
        self.assertTrue(any(h["field"] == "num_classes" and h["value"] == 1001 for h in hints))
        self.assertNotIn("class_names", {h["field"] for h in hints})


if __name__ == "__main__":
    unittest.main()
