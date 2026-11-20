"""Prepare a minimal App Service build context without business data."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

from prepare_nltk import COMMIT as NLTK_COMMIT, extract_archive


ROOT = Path(__file__).resolve().parent.parent
MODEL_REPOSITORY = "https://github.com/ordinarykeys/eaicoding"
MODEL_BASE = "https://raw.githubusercontent.com/ordinarykeys/eaicoding"
NLTK_BASE = "https://raw.githubusercontent.com/nltk/nltk_data"
TIKTOKEN_BASE = "https://openaipublic.blob.core.windows.net/encodings"
NLTK_ARCHIVES = {
    "punkt": {
        "bytes": 13905355,
        "sha256": "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
    },
    "punkt_tab": {
        "bytes": 4319076,
        "sha256": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    },
}


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid relative path: {value!r}")
    return Path(*path.parts)


def verify_file(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"Artifact has unexpected size: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise RuntimeError(f"Artifact hash mismatch: {path.name}")


def acquire_file(destination, expected, local_file=None, url=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if local_file is not None:
        with Path(local_file).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    else:
        count = 0
        with urlopen(url, timeout=120) as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > expected["bytes"]:
                    raise RuntimeError(f"Artifact exceeds expected size: {destination.name}")
                target.write(chunk)
    verify_file(destination, expected)


def copy_context_file(source, destination):
    if source.suffix == ".sh":
        content = source.read_bytes().replace(b"\r\n", b"\n")
        if b"\r" in content:
            raise ValueError(f"Shell script has invalid line endings: {source.name}")
        destination.write_bytes(content)
    else:
        shutil.copyfile(source, destination)


def stage_context(context, model_dir=None, nltk_archives=None, tiktoken_dir=None):
    context = Path(context).resolve()
    if context == ROOT or context.is_relative_to(ROOT):
        raise ValueError("Build context must be outside the application source.")
    if context.exists() and (not context.is_dir() or any(context.iterdir())):
        raise ValueError("Build context must be a new or empty directory.")
    local_inputs = (model_dir, nltk_archives, tiktoken_dir)
    if any(value is not None for value in local_inputs) and any(
        value is None for value in local_inputs
    ):
        raise ValueError("Provide all local model, NLTK archive and tiktoken directories, or none.")
    context.mkdir(parents=True, exist_ok=True)

    files = (ROOT / "environment-setup" / "appservice-context-files.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    for name in files:
        relative = relative_path(name)
        source = ROOT / relative
        if not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Required build-context file is missing or unsafe: {name}")
        destination = context / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_context_file(source, destination)

    manifest = json.loads(
        (ROOT / "environment-setup" / "embedding-download-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (manifest["source_repository"] != MODEL_REPOSITORY
            or not re.fullmatch(r"[a-f0-9]{40}", manifest["source_commit"])):
        raise ValueError("Unrecognized model source or revision.")
    model_path = relative_path(manifest["source_directory"]).as_posix()
    assets = context / ".appservice-assets"
    for item in manifest["files"]:
        path = relative_path(item["path"])
        url = (
            f"{MODEL_BASE}/{manifest['source_commit']}/"
            f"{quote(model_path, safe='/')}/{quote(path.as_posix(), safe='/')}"
        )
        local = Path(model_dir) / path if model_dir is not None else None
        acquire_file(assets / "embedding" / path, item, local_file=local, url=url)
    shutil.copyfile(
        context / "environment-setup" / "bge-model-license.txt",
        assets / "embedding" / "LICENSE.txt",
    )

    with tempfile.TemporaryDirectory(prefix="rfq-nltk-archives-") as directory:
        for name, expected in NLTK_ARCHIVES.items():
            archive = Path(directory) / f"{name}.zip"
            local = Path(nltk_archives) / archive.name if nltk_archives is not None else None
            url = f"{NLTK_BASE}/{NLTK_COMMIT}/packages/tokenizers/{archive.name}"
            acquire_file(archive, expected, local_file=local, url=url)
            extract_archive(archive, assets / "nltk", name)

    tokenizer_manifest = json.loads(
        (ROOT / "environment-setup" / "tiktoken-cache-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if tokenizer_manifest["source_base"] != TIKTOKEN_BASE:
        raise ValueError("Unrecognized tiktoken source.")
    for item in tokenizer_manifest["files"]:
        name = relative_path(item["name"])
        if len(name.parts) != 1:
            raise ValueError("Invalid tiktoken file name.")
        url = f"{TIKTOKEN_BASE}/{quote(name.name, safe='')}"
        cache_key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        if item["cache_key"] != cache_key:
            raise ValueError("Invalid tiktoken cache key.")
        local = Path(tiktoken_dir) / cache_key if tiktoken_dir is not None else None
        acquire_file(
            assets / "tiktoken" / cache_key, item, local_file=local, url=url
        )
    print("Prepared the source-only build context with verified model and tokenizer artifacts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--nltk-archives", type=Path)
    parser.add_argument("--tiktoken-dir", type=Path)
    args = parser.parse_args()
    stage_context(args.context, args.model_dir, args.nltk_archives, args.tiktoken_dir)


if __name__ == "__main__":
    main()
