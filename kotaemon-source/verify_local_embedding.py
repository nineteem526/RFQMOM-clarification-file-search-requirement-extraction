"""Offline smoke test for the downloaded BGE model, not a RAG acceptance test."""

import json
import sys


def block_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
        raise RuntimeError("Network access is forbidden during this verification.")


def main():
    sys.addaudithook(block_network)
    from pilot_runtime import prepare_runtime

    prepare_runtime()
    import numpy as np
    from local_embedding import LocalBGEEmbeddings

    adapter = LocalBGEEmbeddings()

    def embed(text):
        vector = np.array(adapter.invoke(text)[0].embedding)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise RuntimeError("Invalid Kotaemon embedding.")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
            raise RuntimeError("Embedding normalization failed.")
        return vector

    documents = [
        "虚构协议：钢结构平台设计均布荷载为3.0 kN/m²。",
        "虚构协议：电气控制柜防护等级为IP54，控制电源为24 V DC。",
        "虚构协议：钢结构表面防腐采用环氧底漆和聚氨酯面漆。",
    ]
    queries = [
        "钢结构平台的设计荷载是多少？",
        "电气控制柜的防护等级和控制电压是什么？",
        "钢结构防腐涂层有什么要求？",
    ]
    vectors = np.stack([embed(text) for text in documents])
    for expected, query in enumerate(queries):
        vector = embed(query)
        scores = vectors @ vector
        found = int(np.argmax(scores))
        print(json.dumps({
            "case": expected + 1,
            "expected_document": expected + 1,
            "retrieved_document": found + 1,
            "scores": [round(float(score), 4) for score in scores],
            "passed": found == expected,
        }), flush=True)
        if found != expected:
            raise RuntimeError(f"Synthetic retrieval case {expected + 1} failed.")

    print(json.dumps({
        "model_loaded": True,
        "dimension": 512,
        "normalized": True,
        "synthetic_retrieval_cases_passed": 3,
        "network_blocked": True,
        "kotaemon_embedding_adapter_verified": True,
        "business_text_processed": False,
        "kotaemon_retrieval_or_ui_verified": False,
    }), flush=True)


if __name__ == "__main__":
    main()
