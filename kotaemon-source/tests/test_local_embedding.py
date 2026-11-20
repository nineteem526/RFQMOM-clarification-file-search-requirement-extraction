import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from pilot_runtime import prepare_runtime

prepare_runtime()

from kotaemon.base import Document, DocumentWithEmbedding
from theflow.utils.modules import serialize
from local_embedding import BGEEngine, LocalBGEEmbeddings, LocalEmbeddingError, local_engine


class LocalEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LocalBGEEmbeddings()
        cls.engine = local_engine()

    def test_metadata_and_shape(self):
        original = Document(
            content="虚构平台荷载3.0 kN/m²",
            metadata={"filename": "fixture.txt", "location": "paragraph 1"},
        )
        result = self.adapter.invoke(original)[0]
        self.assertIsInstance(result, DocumentWithEmbedding)
        self.assertEqual(result.content, original.content)
        self.assertEqual(result.metadata, original.metadata)
        self.assertEqual(result.id_, original.id_)
        self.assertIsNone(original.embedding)
        self.assertEqual(len(result.embedding), 512)
        self.assertTrue(np.isfinite(result.embedding).all())
        self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=5)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(self.adapter.invoke([]), [])
        for value in ("", "  ", None, 123, ["valid", 42], Document(content=4)):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(LocalEmbeddingError):
                    self.adapter.invoke(value)
        with self.assertRaises(LocalEmbeddingError):
            self.adapter.invoke("valid", unsupported=True)

    def test_token_limit_without_truncation(self):
        self.assertEqual(len(self.engine.encode_input("荷" * 510).ids), 512)
        with self.assertRaises(LocalEmbeddingError):
            self.engine.encode_input("荷" * 511)

    def test_validate_whole_batch_before_inference(self):
        with patch.object(self.engine, "embed_encoded") as inference:
            with self.assertRaises(LocalEmbeddingError):
                self.adapter.invoke(["valid", "荷" * 511])
            inference.assert_not_called()

    def test_async_and_spec(self):
        text = "虚构控制电源24 V DC"
        result = asyncio.run(self.adapter.ainvoke(text))
        self.assertEqual(result[0].embedding, self.adapter.invoke(text)[0].embedding)
        self.assertIn("LocalBGEEmbeddings", str(serialize(self.adapter)))

    def test_hash_mismatch_stops_before_model_load(self):
        with patch("local_embedding.hashlib.file_digest") as digest:
            digest.return_value.hexdigest.return_value = "invalid"
            with patch("local_embedding.ort.InferenceSession") as load:
                with self.assertRaisesRegex(LocalEmbeddingError, "hash mismatch"):
                    BGEEngine()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
