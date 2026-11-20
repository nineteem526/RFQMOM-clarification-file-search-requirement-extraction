"""Runtime paths with backward-compatible Windows defaults and explicit overrides."""

import os
from pathlib import Path
import sys


def _app_service_enabled():
    return os.environ.get("RFQ_APP_SERVICE", "").casefold() in {"1", "true"}


def _app_path():
    value = os.environ.get("APP_PATH")
    if not value:
        raise RuntimeError("APP_PATH is required for the App Service code runtime.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("APP_PATH must be an absolute path.")
    return path


def _app_asset_path(name):
    return _app_path() / "_appservice-assets" / name


def _configured_path(variable):
    value = os.environ.get(variable)
    if value is None:
        return None
    if not value.strip():
        raise ValueError(f"{variable} must not be empty.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{variable} must be an absolute path.")
    return path


def runtime_home():
    configured = _configured_path("RFQ_RUNTIME_HOME")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return Path("/home/rfq-data")
    base = _configured_path("LOCALAPPDATA")
    if base is None:
        raise RuntimeError("Set RFQ_RUNTIME_HOME or LOCALAPPDATA.")
    return base / "RFQKotaemon"


def runtime_venv():
    configured = _configured_path("RFQ_RUNTIME_VENV")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return Path(sys.prefix)
    return runtime_home() / "runtime" / ".venv"


def nltk_data_path():
    configured = _configured_path("RFQ_NLTK_DATA")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return _app_asset_path("nltk")
    return runtime_home() / "resources" / "nltk_data"


def tiktoken_data_path():
    configured = _configured_path("RFQ_TIKTOKEN_DATA")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return _app_asset_path("tiktoken")
    return runtime_home() / "resources" / "tiktoken"


def embedding_model_path(manifest_directory):
    configured = _configured_path("RFQ_EMBEDDING_DIR")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return _app_asset_path("embedding")
    if manifest_directory and "%LOCALAPPDATA%" in manifest_directory:
        suffix = manifest_directory.split("RFQKotaemon", 1)[-1].lstrip("\\/")
        return runtime_home() / Path(suffix.replace("\\", os.sep))
    return Path(os.path.expandvars(manifest_directory))


def ocr_root():
    configured = _configured_path("RFQ_OCR_ROOT")
    if configured is not None:
        return configured
    if _app_service_enabled():
        return _app_asset_path("ocr")
    return runtime_home() / "ocr"


def preview_data_path():
    configured = _configured_path("RFQ_PREVIEW_DATA")
    return configured if configured is not None else runtime_home() / "preview"


def source_scan_path():
    configured = _configured_path("RFQ_SOURCE_SCANS")
    return configured if configured is not None else runtime_home() / "source-scans"
