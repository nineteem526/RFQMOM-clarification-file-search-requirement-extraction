"""Offline OCR subprocess; no company credentials or external inference calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime_paths import ocr_root


def _resolve_ocr_executable() -> Path | None:
    app_service = os.environ.get("RFQ_APP_SERVICE", "").casefold() in {"1", "true"}
    python_directory = "Scripts" if os.name == "nt" else "bin"
    python_executable = "python.exe" if os.name == "nt" else "python"
    executable = ocr_root() / ".venv" / python_directory / python_executable
    if app_service and not executable.is_file():
        executable = Path(sys.executable)
    return executable if executable.is_file() else None


def is_ocr_available() -> bool:
    """Return True when the offline OCR runtime and manifest exist locally."""
    try:
        root = ocr_root()
        executable = _resolve_ocr_executable()
    except (RuntimeError, ValueError):
        return False
    return bool(executable is not None and (root / "model-manifest.json").is_file())


def _run_ocr_worker(worker_args: list[str], timeout_msg: str, timeout: int = 120) -> dict:
    executable = _resolve_ocr_executable()
    if executable is None:
        raise ValueError("本地OCR环境未安装；未使用外部服务替代。")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("DUERR_")
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [str(executable), "-B", str(Path(__file__).resolve()), *worker_args],
            env=environment,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(timeout_msg) from None
    if result.returncode:
        raise ValueError(f"本地OCR失败：{result.stderr[-1800:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError("本地OCR响应格式错误。") from None


def recognize_page(pdf_path, page_number):
    payload = _run_ocr_worker(
        ["--worker", str(Path(pdf_path).resolve()), str(page_number)],
        timeout_msg=f"PDF第{page_number}页本地OCR超时，未标为已识别。",
    )
    if payload.get("page") != page_number or not isinstance(payload.get("lines"), list):
        raise ValueError("本地OCR定位响应不符。")
    return payload["lines"]


def recognize_pdf_region(pdf_path, page_number, rect):
    coords = tuple(float(v) for v in rect)[:4]
    if len(coords) != 4:
        raise ValueError("PDF 区域坐标无效。")
    payload = _run_ocr_worker(
        [
            "--worker-region",
            str(Path(pdf_path).resolve()),
            str(page_number),
            *(f"{v:.2f}" for v in coords),
        ],
        timeout_msg=f"PDF第{page_number}页插图区域本地OCR超时，未标为已识别。",
    )
    if payload.get("page") != page_number or not isinstance(payload.get("lines"), list):
        raise ValueError("本地OCR定位响应不符。")
    return payload["lines"]


def recognize_image(image_path):
    payload = _run_ocr_worker(
        ["--worker-image", str(Path(image_path).resolve())],
        timeout_msg="图片本地OCR超时，未标为已识别。",
    )
    if not isinstance(payload.get("pages"), list):
        raise ValueError("本地OCR图片响应格式不符。")
    return payload["pages"]


def _init_worker_engine(require_fitz: bool = True):
    def offline(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise RuntimeError("OCR worker forbids all network connections and DNS.")

    sys.addaudithook(offline)
    if require_fitz:
        import fitz  # noqa: F401
    import numpy as np  # noqa: F401

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
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


def _extract_lines(engine, bgr_pixels, scale: float, offset_x: float = 0.0, offset_y: float = 0.0) -> list[dict]:
    result, _ = engine(bgr_pixels)
    lines = []
    for box, text, score in result or []:
        coordinates = [
            [round(float(x) / scale + offset_x, 2), round(float(y) / scale + offset_y, 2)]
            for x, y in box
        ]
        lines.append({"text": text, "confidence": float(score), "box": coordinates})
    return lines


def worker(path, page_number):
    engine = _init_worker_engine(require_fitz=True)
    import fitz
    import numpy as np

    with fitz.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("OCR页码超出原件范围。")
        page = pdf[page_number - 1]
        scale = min(2.0, 2400 / max(page.rect.width, page.rect.height, 1))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
        lines = _extract_lines(engine, pixels[:, :, ::-1], scale)
    print(json.dumps({"page": page_number, "lines": lines}, ensure_ascii=False))


def worker_region(path, page_number, x0, y0, x1, y1):
    engine = _init_worker_engine(require_fitz=True)
    import fitz
    import numpy as np

    with fitz.open(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("OCR页码超出原件范围。")
        page = pdf[page_number - 1]
        clip = fitz.Rect(x0, y0, x1, y1) & page.rect
        if clip.is_empty or clip.width <= 1 or clip.height <= 1:
            print(json.dumps({"page": page_number, "lines": []}, ensure_ascii=False))
            return
        scale = min(2.0, 2000 / max(clip.width, clip.height, 1))
        pixmap = page.get_pixmap(clip=clip, matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )
        lines = _extract_lines(engine, pixels[:, :, ::-1], scale, offset_x=clip.x0, offset_y=clip.y0)
    print(json.dumps({"page": page_number, "lines": lines}, ensure_ascii=False))


def worker_image(path):
    engine = _init_worker_engine(require_fitz=False)
    import numpy as np
    from PIL import Image, ImageOps

    pages_out = []
    with Image.open(path) as img:
        n_frames = min(int(getattr(img, "n_frames", 1) or 1), 20)
        for frame_idx in range(n_frames):
            img.seek(frame_idx)
            frame = ImageOps.exif_transpose(img).convert("RGB")
            w, h = frame.size
            scale = min(1.0, 2400 / max(w, h, 1))
            if scale < 1.0:
                frame = frame.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.Resampling.LANCZOS,
                )
            pixels = np.asarray(frame, dtype=np.uint8)
            lines = _extract_lines(engine, pixels[:, :, ::-1], scale)
            pages_out.append({"page": frame_idx + 1, "lines": lines})
    print(json.dumps({"pages": pages_out}, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--worker" and len(sys.argv) == 4:
        worker(sys.argv[2], int(sys.argv[3]))
    elif mode == "--worker-region" and len(sys.argv) == 8:
        worker_region(
            sys.argv[2],
            int(sys.argv[3]),
            float(sys.argv[4]),
            float(sys.argv[5]),
            float(sys.argv[6]),
            float(sys.argv[7]),
        )
    elif mode == "--worker-image" and len(sys.argv) == 3:
        worker_image(sys.argv[2])
    else:
        raise SystemExit("Only the local OCR worker entry points are supported.")
