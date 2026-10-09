import json
import os


class LLMError(RuntimeError):
    pass


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def merge_system(messages):
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest and rest[0]["role"] == "user":
        rest[0]["content"] = "\n\n".join(system) + "\n\n" + rest[0]["content"]
    return rest


class StructuredLLM:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def _complete(self, messages, schema):
        raise NotImplementedError

    def generate_text(self, system, user, history=()):
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]
        text = (self._complete(messages, None) or "").strip()
        if not text:
            raise LLMError("empty reply")
        return text

    def generate_structured(self, system, user, schema, check=None):
        schema_json = schema.model_json_schema()
        messages = [
            {"role": "system", "content": system + "\n\nReply with one JSON object matching this schema and nothing else:\n" + json.dumps(schema_json)},
            {"role": "user", "content": user},
        ]
        last = None
        for _ in range(self.max_retries + 1):
            text = self._complete(messages, schema_json)
            try:
                obj = schema.model_validate_json(extract_json(text))
                problem = check(obj) if check else None
                if problem:
                    raise ValueError(problem)
                return obj
            except ValueError as exc:
                last = exc
                note = str(exc).replace("\n", " ")[:300]
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That reply was rejected: {note}. Reply again with corrected JSON only."},
                ]
        raise LLMError(f"no valid reply after {self.max_retries + 1} attempts: {str(last)[:200]}")


class NoLLM(StructuredLLM):
    def _complete(self, messages, schema):
        raise LLMError("no LLM backend configured (MODELLAB_LLM_BACKEND=none)")


class ScriptedLLM(StructuredLLM):
    def __init__(self, replies, max_retries=2):
        super().__init__(max_retries)
        self.replies = list(replies)
        self.calls = []

    def _complete(self, messages, schema):
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("scripted replies exhausted")
        return self.replies.pop(0)


class LlamaCppLLM(StructuredLLM):
    def __init__(self, model_path=None, repo_id=None, filename=None, n_ctx=4096, n_threads=None,
                 n_gpu_layers=0, temperature=0.2, max_tokens=600, max_retries=2):
        super().__init__(max_retries)
        from llama_cpp import Llama

        self.temperature, self.max_tokens = temperature, max_tokens
        opts = dict(n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers, verbose=False)
        if model_path:
            self.llm = Llama(model_path=model_path, **opts)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **opts)
        else:
            raise LLMError("set MODELLAB_LLM_PATH or both MODELLAB_LLM_REPO and MODELLAB_LLM_FILE")

    def _complete(self, messages, schema):
        extra = {"response_format": {"type": "json_object", "schema": schema}} if schema else {}
        out = self.llm.create_chat_completion(
            messages=merge_system(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra,
        )
        return out["choices"][0]["message"]["content"] or ""


class OpenAICompatLLM(StructuredLLM):
    def __init__(self, base_url, model, api_key="not-needed", temperature=0.2, max_tokens=600, max_retries=2):
        super().__init__(max_retries)
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def _complete(self, messages, schema):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content or ""


def provider_from_env(env=None):
    env = os.environ if env is None else env
    backend = env.get("MODELLAB_LLM_BACKEND", "llama_cpp")
    if backend == "none":
        return NoLLM()
    if backend == "llama_cpp":
        return LlamaCppLLM(
            model_path=env.get("MODELLAB_LLM_PATH"),
            repo_id=env.get("MODELLAB_LLM_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
            filename=env.get("MODELLAB_LLM_FILE", "*q4_k_m.gguf"),
            n_ctx=int(env.get("MODELLAB_LLM_CTX", "4096")),
            n_threads=int(env["MODELLAB_LLM_THREADS"]) if env.get("MODELLAB_LLM_THREADS") else None,
            n_gpu_layers=int(env.get("MODELLAB_LLM_GPU_LAYERS", "0")),
        )
    if backend == "openai_compat":
        if not env.get("MODELLAB_LLM_BASE_URL") or not env.get("MODELLAB_LLM_MODEL"):
            raise LLMError("openai_compat needs MODELLAB_LLM_BASE_URL and MODELLAB_LLM_MODEL")
        return OpenAICompatLLM(
            base_url=env["MODELLAB_LLM_BASE_URL"],
            model=env["MODELLAB_LLM_MODEL"],
            api_key=env.get("MODELLAB_LLM_API_KEY", "not-needed"),
        )
    raise LLMError(f"unknown MODELLAB_LLM_BACKEND {backend!r}")
