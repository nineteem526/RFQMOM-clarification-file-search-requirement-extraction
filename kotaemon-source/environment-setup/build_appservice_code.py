"""Build a source-only Linux App Service ZIP from verified local runtime assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid
import zipfile

from stage_appservice import ROOT, acquire_file, copy_context_file, relative_path, stage_context


APP_FILES = frozenset({".py", ".css", ".js"})
APP_MANIFESTS = frozenset({
    "environment-setup/embedding-download-manifest.json",
    "environment-setup/tiktoken-cache-manifest.json",
})
UPSTREAM_REVISION = "9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9"
REQUIRED_ICONS = ("dark_mode", "expand", "rename", "delete", "new")


def _copy_tree(source, destination):
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"Expected an asset directory: {source}")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _write_requirements(package):
    requirements = package / "requirements.txt"
    lines = ["-c upstream-constraints.txt"]
    lines.extend(
        f"{name} @ git+https://github.com/Cinnamon/kotaemon.git@"
        f"{UPSTREAM_REVISION}#subdirectory=libs/{name}"
        for name in ("kotaemon", "ktem")
    )
    for relative in (
        "environment-setup/required-dependencies.in",
        "environment-setup/appservice-runtime-requirements.txt",
        "environment-setup/ocr-linux-requirements.txt",
    ):
        source = ROOT / relative
        lines.extend(source.read_text(encoding="utf-8").splitlines())
    requirements.write_text(
        "\n".join(line.rstrip() for line in lines if line.strip()) + "\n",
        encoding="utf-8",
    )


def _stage_package(context, package, icons_dir, stopwords_dir):
    entries = (ROOT / "environment-setup" / "appservice-context-files.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    for entry in entries:
        source = context / Path(entry)
        if entry in APP_MANIFESTS:
            destination = package / Path(entry)
        elif source.suffix.lower() in APP_FILES:
            destination = package / source.name
        else:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_context_file(source, destination)
    for entry in APP_MANIFESTS:
        if not (package / Path(entry)).is_file():
            raise RuntimeError(f"Required application manifest is missing: {entry}")

    shutil.copyfile(context / "environment-setup" / "upstream-constraints.txt",
                    package / "upstream-constraints.txt")
    _write_requirements(package)
    copy_context_file(ROOT / "start-appservice.sh", package / "start-appservice.sh")
    shutil.copytree(context / ".appservice-assets", package / "_appservice-assets")

    assets = package / "_appservice-assets"
    (assets / "ocr").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "environment-setup" / "ocr-model-manifest.json",
        assets / "ocr" / "model-manifest.json",
    )
    for icon in REQUIRED_ICONS:
        path = Path(icons_dir) / f"{icon}.svg"
        if not path.is_file():
            raise RuntimeError(f"Required Kotaemon icon is missing: {icon}")
    _copy_tree(icons_dir, assets / "icons")
    _copy_tree(stopwords_dir, assets / "nltk" / "corpora" / "stopwords")

    for required in ("tokenizers/punkt", "tokenizers/punkt_tab/english",
                     "corpora/stopwords/english"):
        if not (assets / "nltk" / required).exists():
            raise RuntimeError(f"Required offline NLTK resource is missing: {required}")
    manifest = ROOT / "environment-setup" / "tiktoken-cache-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for required in ("cl100k_base.tiktoken", "o200k_base.tiktoken"):
        match = next(item for item in data["files"] if item["name"] == required)
        path = assets / "tiktoken" / match["cache_key"]
        if not path.is_file() or path.stat().st_size != match["bytes"]:
            raise RuntimeError(f"Required offline tokenizer is missing or incomplete: {required}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != match["sha256"]:
            raise RuntimeError(f"Required offline tokenizer has the wrong hash: {required}")


def _stage_headless_opencv(assets, local_wheel=None):
    manifest_path = ROOT / "environment-setup" / "opencv-headless-manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_roots = {
        "cv2", "opencv_python_headless.libs", "opencv_python_headless-4.10.0.84.dist-info",
    }
    destination = assets / "ocr" / "opencv-headless"
    with tempfile.TemporaryDirectory(prefix="rfq-opencv-wheel-") as directory:
        wheel = Path(directory) / expected["filename"]
        acquire_file(wheel, expected, local_file=local_wheel, url=expected["url"])
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                relative = relative_path(entry.filename)
                if (relative.parts[0] not in allowed_roots
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("OpenCV wheel contains an unsafe entry.")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    if not (destination / "cv2" / "cv2.abi3.so").is_file():
        raise RuntimeError("OpenCV wheel is missing the Linux native module.")
    if not (destination / "cv2" / "LICENSE-3RD-PARTY.txt").is_file():
        raise RuntimeError("OpenCV wheel is missing its third-party license notices.")
    shutil.copyfile(manifest_path, assets / "ocr" / "opencv-headless-manifest.json")


def build_package(output, model_dir, nltk_archives, tiktoken_dir, icons_dir,
                  stopwords_dir, opencv_headless_wheel=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing deployment package: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Deployment package path must end in .zip.")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rfq-appservice-code-") as directory:
        temporary = Path(directory)
        context = temporary / "context"
        package = temporary / "package"
        package.mkdir()
        stage_context(context, model_dir, nltk_archives, tiktoken_dir)
        _stage_package(context, package, icons_dir, stopwords_dir)
        _stage_headless_opencv(package / "_appservice-assets", opencv_headless_wheel)

        partial = output.with_name(output.name + f".{uuid.uuid4().hex}.partial")
        try:
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                for path in sorted(package.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(package).as_posix())
            os.link(partial, output)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Built source-only App Service package: {output}")


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "model-dir", "nltk-archives", "tiktoken-dir", "icons-dir",
                 "stopwords-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--opencv-headless-wheel", type=Path)
    args = parser.parse_args()
    build_package(
        args.output, args.model_dir, args.nltk_archives, args.tiktoken_dir,
        args.icons_dir, args.stopwords_dir, args.opencv_headless_wheel,
    )


if __name__ == "__main__":
    main()
