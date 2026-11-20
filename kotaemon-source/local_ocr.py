"""Offline OCR subprocess; no company credentials or external inference calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime_paths import ocr_root


def recognize_page(pdf_path, page_number):
    app_service = os.environ.get("RFQ_APP_SERVICE", "").casefold() in {"1", "true"}
    python_directory = "Scripts" if os.name == "nt" else "bin"
    python_executable = "python.exe" if os.name == "nt" else "python"
    executable = ocr_root() / ".venv" / python_directory / python_executable
    if app_service and not executable.is_file():
        executable = Path(sys.executable)
    if not executable.is_file():
        raise ValueError("本地OCR环境未安装；未使用外部服务替代。")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("DUERR_")}
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [str(executable), "-B", str(Path(__file__).resolve()),
             "--worker", str(Path(pdf_path).resolve()), str(page_number)],
            env=environment, capture_output=True, encoding="utf-8", timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"PDF第{page_number}页本地OCR超时，未标为已识别。") from None
    if result.returncode:
        raise ValueError(f"本地OCR失败：{result.stderr[-1800:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError("本地OCR响应格式错误。") from None
    if payload.get("page") != page_number or not isinstance(payload.get("lines"), list):
        raise ValueError("本地OCR定位响应不符。")
    return payload["lines"]


def worker(path, page_number):
    def offline(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise RuntimeError("OCR worker forbids all network connections and DNS.")
    sys.addaudithook(offline)
    import fitz
    import numpy as np

    headless = os.environ.get("RFQ_OPENCV_HEADLESS_DIR")
    if headless:
        directory = Path(headless)
        if not directory.is_absolute() or not (directory / "cv2").is_dir():
            raise ValueError("App Service headless OpenCV assets are missing or invalid.")
        sys.path.insert(0, str(directory))
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR

    manifest = json.loads((ocr_root() / "model-manifest.json").read_text(encoding="utf-8"))
    package = Path(rapidocr_onnxruntime.__file__).parent
    for name, expected in manifest["models"].items():
        file = package / "models" / name
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("OCR模型哈希不符，拒绝识别。")
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    with fitz.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("OCR页码超出原件范围。")
        page = pdf[page_number - 1]
        scale = min(2.0, 2400 / max(page.rect.width, page.rect.height))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
        result, _ = engine(pixels[:, :, ::-1])
        lines = []
        for box, text, score in result or []:
            coordinates = [[round(float(x) / scale, 2), round(float(y) / scale, 2)]
                           for x, y in box]
            lines.append({"text": text, "confidence": float(score), "box": coordinates})
    print(json.dumps({"page": page_number, "lines": lines}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Only the local OCR worker entry point is supported.")
    worker(sys.argv[2], int(sys.argv[3]))
