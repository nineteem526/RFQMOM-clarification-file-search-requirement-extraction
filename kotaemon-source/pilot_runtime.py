"""Explicit local runtime preparation, before importing Kotaemon."""

import os
from pathlib import Path
import sys

from runtime_paths import nltk_data_path, runtime_home, runtime_venv, tiktoken_data_path


def prepare_runtime():
    root = runtime_home()
    root.mkdir(parents=True, exist_ok=True)
    expected = runtime_venv()
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError("Use the configured RFQKotaemon virtual environment.")
    resources = nltk_data_path()
    for name in ("tokenizers/punkt", "tokenizers/punkt_tab/english",
                 "corpora/stopwords/english"):
        if not (resources / name).exists():
            raise RuntimeError("Local NLTK resource missing; run prepare_nltk.py.")
    os.environ.update({
        "NLTK_DATA": str(resources),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "ANONYMIZED_TELEMETRY": "False",
        "HAYSTACK_TELEMETRY_ENABLED": "False",
        "OTEL_SDK_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "TIKTOKEN_CACHE_DIR": str(tiktoken_data_path()),
    })
    import nltk

    def reject_download(*args, **kwargs):
        raise RuntimeError("Automatic NLTK downloads are disabled.")

    nltk.download = reject_download
    return root
