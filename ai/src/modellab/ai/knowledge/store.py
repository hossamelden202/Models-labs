import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modellab.ai.knowledge.documents import KnowledgeDocument


class HashingEmbedder:
    def __init__(self, dim=384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                h = zlib.crc32(token.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device="cpu")

    def embed(self, texts):
        vecs = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@dataclass
class SearchHit:
    doc: KnowledgeDocument
    score: float


def _text(doc):
    return f"{doc.title}\n{doc.content}"


class KnowledgeStore:
    def __init__(self, embedder, path=None):
        self.embedder = embedder
        self.path = Path(path) if path else None
        self.docs = {}
        self.vecs = {}
        if self.path:
            self.load()

    def __len__(self):
        return len(self.docs)

    def add(self, docs):
        fresh = [d for d in docs if self.docs.get(d.id) != d]
        if not fresh:
            return 0
        vectors = self.embedder.embed([_text(d) for d in fresh])
        for doc, vec in zip(fresh, vectors):
            self.docs[doc.id] = doc
            self.vecs[doc.id] = vec
        return len(fresh)

    def search(self, query, top_k=4, source_type=None, filters=None):
        ids = [
            i for i, d in self.docs.items()
            if (source_type is None or d.source_type == source_type)
            and all(d.metadata.get(k) == v for k, v in (filters or {}).items())
        ]
        if not ids:
            return []
        q = self.embedder.embed([query])[0]
        scores = np.stack([self.vecs[i] for i in ids]) @ q
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(self.docs[ids[j]], float(scores[j])) for j in order]

    def save(self):
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        ids = list(self.docs)
        payload = {"embedder": self.embedder.name, "ids": ids, "docs": [self.docs[i].model_dump() for i in ids]}
        (self.path / "index.json").write_text(json.dumps(payload, indent=1))
        if ids:
            np.save(self.path / "vectors.npy", np.stack([self.vecs[i] for i in ids]))

    def load(self):
        index = self.path / "index.json"
        if not index.exists():
            return
        payload = json.loads(index.read_text())
        docs = [KnowledgeDocument.model_validate(d) for d in payload["docs"]]
        vectors = self.path / "vectors.npy"
        if payload.get("embedder") == self.embedder.name and vectors.exists() and docs:
            matrix = np.load(vectors)
            if len(matrix) == len(docs):
                self.docs = {d.id: d for d in docs}
                self.vecs = {d.id: matrix[i] for i, d in enumerate(docs)}
                return
        self.add(docs)
