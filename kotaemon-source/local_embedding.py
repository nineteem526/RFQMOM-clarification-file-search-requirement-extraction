"""Local-only, hash-checked BGE adapter for Kotaemon."""

import asyncio
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from runtime_paths import embedding_model_path
from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.embeddings.base import BaseEmbeddings


class LocalEmbeddingError(ValueError):
    pass


class BGEEngine:
    def __init__(self):
        manifest_path = (
            Path(__file__).resolve().parent
            / "environment-setup"
            / "embedding-download-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_dir = embedding_model_path(manifest["local_directory"])
        for item in manifest["files"]:
            path = model_dir.joinpath(*item["path"].split("/"))
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise LocalEmbeddingError(f"Missing or invalid artifact: {item['path']}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise LocalEmbeddingError(f"Artifact hash mismatch: {item['path']}")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if {node.name for node in self.session.get_inputs()} != {
            "input_ids", "attention_mask", "token_type_ids"
        }:
            raise LocalEmbeddingError("Unexpected model inputs.")
        if "last_hidden_state" not in {
            node.name for node in self.session.get_outputs()
        }:
            raise LocalEmbeddingError("Model has no last_hidden_state output.")

    def encode_input(self, text):
        if not isinstance(text, str) or not text.strip():
            raise LocalEmbeddingError("Embedding input must be non-empty text.")
        encoded = self.tokenizer.encode(text)
        if len(encoded.ids) > 512:
            raise LocalEmbeddingError(
                f"Input has {len(encoded.ids)} tokens; limit is 512. "
                "Split the source into smaller traceable chunks before indexing."
            )
        if not encoded.ids or encoded.ids[0] != self.tokenizer.token_to_id("[CLS]"):
            raise LocalEmbeddingError("Tokenizer did not prepend CLS.")
        return encoded

    def embed_encoded(self, encoded):
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }
        hidden = self.session.run(["last_hidden_state"], feed)[0]
        if hidden.shape != (1, len(encoded.ids), 512):
            raise LocalEmbeddingError(f"Unexpected model output shape: {hidden.shape}")
        # BGE uses CLS pooling, not mean pooling.
        vector = hidden[0, 0, :]
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise LocalEmbeddingError("Embedding contains invalid values.")
        return (vector / norm).tolist()


@lru_cache(maxsize=1)
def local_engine():
    return BGEEngine()


class LocalBGEEmbeddings(BaseEmbeddings):
    """Symmetric encoding: no instruction is added to document or query text."""

    def invoke(self, text, *args, **kwargs):
        if args or kwargs:
            raise LocalEmbeddingError("Unsupported embedding parameters.")
        if isinstance(text, (str, Document)):
            items = [text]
        elif isinstance(text, list):
            items = text
        else:
            raise LocalEmbeddingError("Expected text, Document, or a list of either.")
        if not items:
            return []
        if any(not isinstance(item, (str, Document)) for item in items):
            raise LocalEmbeddingError("All embedding items must be text or Documents.")
        documents = [Document(content=item) for item in items]
        engine = local_engine()
        encoded = [engine.encode_input(doc.content) for doc in documents]
        return [
            DocumentWithEmbedding(content=doc, embedding=engine.embed_encoded(tokens))
            for doc, tokens in zip(documents, encoded)
        ]

    async def ainvoke(self, text, *args, **kwargs):
        return await asyncio.to_thread(self.invoke, text, *args, **kwargs)
