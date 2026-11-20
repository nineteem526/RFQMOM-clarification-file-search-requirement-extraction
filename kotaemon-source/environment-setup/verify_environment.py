"""Offline import checks only; never initializes the Kotaemon application."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime_paths import nltk_data_path, runtime_venv


def forbid_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("NETWORK_BLOCKED_DURING_ENVIRONMENT_CHECK")
    if event == "socket.bind" and args[1] not in {("::1", 0), ("127.0.0.1", 0)}:
        raise RuntimeError("LISTENER_BLOCKED_DURING_ENVIRONMENT_CHECK")


def main():
    expected = runtime_venv()
    if Path(sys.prefix).resolve() != expected.resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with the dedicated Kotaemon virtual environment.")
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("Python 3.11 x64 required.")
    os.environ.update({
        "NLTK_DATA": str(nltk_data_path()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "ANONYMIZED_TELEMETRY": "False",
        "HAYSTACK_TELEMETRY_ENABLED": "False",
        "OTEL_SDK_DISABLED": "true",
    })
    sys.addaudithook(forbid_network)
    import nltk
    def reject_download(*args, **kwargs):
        raise RuntimeError("AUTOMATIC_NLTK_DOWNLOAD_BLOCKED")
    nltk.download = reject_download
    failures = []
    for name in (
        "numpy", "hnswlib", "onnxruntime", "chromadb", "pypdf", "docx",
        "openpyxl", "gradio", "openai", "kotaemon", "ktem", "kotaemon.llms",
    ):
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append({"module": name, "error_type": type(error).__name__,
                             "message": str(error)})
            print(json.dumps({"module": name, "status": "failed",
                              "error_type": type(error).__name__}))
        else:
            print(json.dumps({"module": name, "status": "passed"}))
    print(json.dumps({
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": {name: metadata.version(name) for name in ("kotaemon", "ktem")},
        "failures": failures,
        "application_started": False,
        "network_blocked": True,
    }))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
