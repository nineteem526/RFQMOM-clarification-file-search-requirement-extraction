"""Bounded local Office conversion of snapshots, never live business files."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert_snapshot(source, timeout=90):
    source = Path(source).resolve(strict=True)
    suffix = source.suffix.lower()
    if suffix not in {".doc", ".wps", ".xls"}:
        raise ValueError("Only legacy document snapshots may be converted.")
    if sys.platform != "win32":
        raise ValueError(
            f"{suffix}旧格式转换需要Windows版Microsoft Office；"
            "当前Linux运行环境不会尝试转换。"
        )
    with source.open("rb") as stream:
        signature = stream.read(128).lstrip()
    if not signature.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04", b"{\\rtf")) and not (
        signature.lower().startswith((b"<html", b"<!doctype", b"<?xml"))
    ):
        raise ValueError("旧格式文件头无法识别；未启动Office，也未作为成功资料录入。")
    target = source.parent / ("converted-" + uuid.uuid4().hex + (
        ".xlsx" if suffix == ".xls" else ".docx"
    ))
    receipt = target.with_suffix(".conversion.json")
    lease = target.with_suffix(".process.json")
    original_hash = file_hash(source)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
               str(source), str(target), str(lease)]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, creationflags=subprocess.CREATE_NO_WINDOW,
            env={**{key: value for key, value in os.environ.items()
                    if not key.upper().startswith("DUERR_")},
                 "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        # The worker records only a fresh DispatchEx process, never an existing user's PID.
        if lease.exists():
            import psutil

            owner = json.loads(lease.read_text(encoding="utf-8"))
            try:
                process = psutil.Process(owner["pid"])
                if process.create_time() == owner["created"]:
                    process.terminate()
                    process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
        raise ValueError("本地Office转换超时；未进入索引，请关闭弹出的转换提示后重试。") from None
    finally:
        if lease.exists():
            lease.unlink()
    if completed.returncode or not target.is_file():
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-6000:]
        raise ValueError("本地Office转换失败：" + (detail or "未生成转换副本"))
    if file_hash(source) != original_hash:
        raise ValueError("原件快照在转换期间发生变化，拒绝索引。")
    record = {
        "source": source.name, "source_sha256": original_hash,
        "converted": target.name, "converted_sha256": file_hash(target),
        "converter": "installed Microsoft Excel" if suffix == ".xls" else "installed Microsoft Word",
        "macro_execution": "force disabled", "read_only_source": True,
        "date_policy": "Use original source metadata, never converted file creation time.",
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return target, record


def worker(source, target, lease):
    import pythoncom
    import psutil
    import win32com.client
    import win32process

    pythoncom.CoInitialize()
    application = document = scratch = None
    owned = False
    existing = set(psutil.pids())
    saved = {}
    excel = source.suffix.lower() in {".xls", ".xlsx"}
    try:
        application = win32com.client.DispatchEx(
            "Excel.Application" if excel else "Word.Application"
        )
        if excel:
            # Excel requires an open workbook before Calculation can be changed.
            scratch = application.Workbooks.Add()
            _, pid = win32process.GetWindowThreadProcessId(application.Hwnd)
        else:
            fresh = [item.pid for item in psutil.process_iter(["name"])
                     if item.pid not in existing
                     and (item.info["name"] or "").casefold() == "winword.exe"]
            if len(fresh) != 1:
                raise RuntimeError("不能唯一确定独立Word进程，拒绝操作现有文档。")
            pid = fresh[0]
        if pid in existing:
            raise RuntimeError("Office复用了已有进程；为保护用户文档，拒绝操作。")
        process = psutil.Process(pid)
        owned = True
        lease.write_text(json.dumps({"pid": pid, "created": process.create_time()}),
                         encoding="utf-8")
        for attribute, value in (
            ("Visible", False), ("DisplayAlerts", False if excel else 0),
            ("AutomationSecurity", 3),
        ):
            saved[attribute] = getattr(application, attribute)
            setattr(application, attribute, value)
        if excel:
            for attribute, value in (
                ("EnableEvents", False), ("AskToUpdateLinks", False),
                ("Calculation", -4135), ("CalculateBeforeSave", False),
            ):
                saved[attribute] = getattr(application, attribute)
                setattr(application, attribute, value)
            document = application.Workbooks.Open(
                str(source), UpdateLinks=0, ReadOnly=True,
                Password=uuid.uuid4().hex, WriteResPassword=uuid.uuid4().hex,
                IgnoreReadOnlyRecommended=True, AddToMru=False, Notify=False,
            )
            document.CheckCompatibility = False
            document.SaveAs(str(target), FileFormat=56 if target.suffix == ".xls" else 51,
                            AddToMru=False)
        else:
            for option in ("UpdateLinksAtOpen", "UpdateFieldsAtPrint", "UpdateLinksAtPrint"):
                saved["Options." + option] = getattr(application.Options, option)
                setattr(application.Options, option, False)
            document = application.Documents.Open(
                str(source), ConfirmConversions=False, ReadOnly=True,
                AddToRecentFiles=False, PasswordDocument=uuid.uuid4().hex,
                WritePasswordDocument=uuid.uuid4().hex, Visible=False,
                OpenAndRepair=False, NoEncodingDialog=True,
            )
            document.SaveAs2(str(target), FileFormat=0 if target.suffix == ".doc" else 12,
                             AddToRecentFiles=False)
    finally:
        try:
            if application is not None and owned:
                try:
                    for attribute, value in saved.items():
                        if attribute.startswith("Options."):
                            setattr(application.Options, attribute.split(".")[1], value)
                        else:
                            setattr(application, attribute, value)
                finally:
                    try:
                        if document is not None:
                            document.Close(SaveChanges=False)
                    finally:
                        application.Quit()
        finally:
            pythoncom.CoUninitialize()


if __name__ == "__main__":
    try:
        worker(*(Path(item) for item in sys.argv[2:5]))
    except Exception as error:
        # No source contents or credentials in conversion diagnostics.
        import traceback

        traceback.print_exc(limit=8)
        sys.exit(1)
