"""Install previously reviewed, checksum-verified archives without networking."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime_paths import nltk_data_path, runtime_home, runtime_venv


COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"


def extract_archive(archive, target, name):
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != name
                    or any(":" in p or "\\" in p for p in path.parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError(f"Unsafe archive member in {name}")
            destination = target / "tokenizers" / Path(*path.parts)
            if member.is_dir():
                continue
            content = zipped.read(member)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise RuntimeError(f"Existing resource differs: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(content)


def main():
    root = runtime_home()
    runtime = root / "runtime"
    if Path(sys.prefix).resolve() != runtime_venv().resolve():
        raise RuntimeError("Dedicated virtual environment required.")
    downloads = runtime / "downloads" / "nltk"
    target = nltk_data_path()
    index = ET.parse(downloads / "index.xml").getroot()
    packages = {}
    for name in ("punkt", "punkt_tab"):
        entry = index.find(f"./packages/package[@id='{name}']")
        if entry is None:
            raise RuntimeError(f"Missing index entry: {name}")
        archive = downloads / f"{name}.zip"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != entry.attrib["sha256_checksum"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        packages[name] = {"sha256": digest, "archive": str(archive)}
        extract_archive(archive, target, name)
    package = importlib.metadata.distribution("llama-index-core")
    bundled = Path(package.locate_file(
        "llama_index/core/_static/nltk_cache/corpora/stopwords"
    ))
    for source in bundled.rglob("*"):
        if source.is_file():
            destination = target / "corpora" / "stopwords" / source.relative_to(bundled)
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise RuntimeError("Existing stopwords resource differs.")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    if not (target / "corpora" / "stopwords" / "english").is_file():
        raise RuntimeError("Bundled stopwords missing.")
    manifest = {
        "repository": "https://github.com/nltk/nltk_data",
        "commit": COMMIT,
        "packages": packages,
        "stopwords_source": f"Bundled llama-index-core {package.version}",
        "license_status": "Unclarified; user directed proceeding for local personal trial.",
        "network_used": False,
        "archive_pickle_models_executed": False,
    }
    (target / "resource-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Verified NLTK resources available at: {target}")


if __name__ == "__main__":
    main()
