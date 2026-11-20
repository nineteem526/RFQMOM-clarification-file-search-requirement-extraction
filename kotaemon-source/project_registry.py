"""Project discovery and creation restricted to the single authorized source."""

from contextlib import closing
from pathlib import Path
import sqlite3
import json
import os
import subprocess
import sys
import time
import urllib.request
import threading

_service_operations = threading.Lock()


SOURCE_ENV = "RFQ_AUTHORIZED_SOURCE"


def authorized_source():
    # Overriding is an authorization decision; existing projects stay bound to their original source.
    configured = os.environ.get(SOURCE_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "评审后"


def projects(data_root):
    result = []
    root = Path(data_root) / "projects"
    for path in sorted(root.glob("*/project.sqlite3")):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT value FROM state WHERE key='source'").fetchone()
        if row and Path(row[0]).resolve() == authorized_source().resolve():
            result.append(path.parent.name)
    return result


def create_project(data_root, identity):
    from project_workspace import ProjectWorkspace

    target = Path(data_root) / "projects" / identity
    if target.exists():
        raise ValueError("项目ID已存在，未覆盖。")
    # Validation runs before any directory is created by ProjectWorkspace.
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", identity):
        raise ValueError("项目ID限小写字母、数字、短横线，最多48字符。")
    workspace = ProjectWorkspace(data_root, identity, authorized_source())
    return workspace.project_id


def open_project(data_root, identity):
    with _service_operations:
        return _open_project(data_root, identity)


def _recorded_process(info, identity):
    import psutil

    if not isinstance(info.get("pid"), int) or info["pid"] <= 0:
        raise ValueError("项目服务进程记录无效。")
    try:
        process = psutil.Process(info["pid"])
        command = process.cmdline()
        executables = {Path(sys.executable).resolve(), Path(sys._base_executable).resolve()}
        if (Path(process.exe()).resolve() not in executables
                or str(Path(__file__).parent / "project_app.py") not in command
                or "--project" not in command
                or command[command.index("--project") + 1] != identity
                or "--port" not in command
                or command[command.index("--port") + 1] != str(info["port"])):
            raise ValueError("进程身份不匹配，未操作其他服务。")
        return process
    except psutil.NoSuchProcess:
        return None


def _open_project(data_root, identity):
    if identity not in projects(data_root):
        raise ValueError("项目不存在或源目录不在授权范围。")
    root = Path(data_root) / "projects" / identity
    info_path = root / "local-service.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        port = info["port"]
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("项目服务端口记录无效。")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/project-health", timeout=2) as response:
                health = json.load(response)
            if health.get("app") == "rfq-kotaemon-project" and health.get("project") == identity:
                return f"http://127.0.0.1:{port}"
        except (OSError, ValueError):
            pass  # Stopped services are restarted below, not reported as available.
        if _recorded_process(info, identity) is not None:
            raise ValueError("该项目进程尚在运行但入口未就绪；请稍后再试或停止该项目，未重复启动。")
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log_path = root / "service-output.log"
    with log_path.open("ab") as log:
        process_options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).parent / "project_app.py"),
             "--project", identity, "--port", str(port)],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            env=os.environ.copy(), **process_options,
        )
    info_path.write_text(json.dumps({"port": port, "pid": process.pid}), encoding="utf-8")
    for _ in range(90):
        if process.poll() is not None:
            raise ValueError(f"项目服务启动失败，请查看{log_path}。")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/project-health", timeout=1) as response:
                health = json.load(response)
            if health.get("app") == "rfq-kotaemon-project" and health.get("project") == identity:
                return f"http://127.0.0.1:{port}"
        except (OSError, ValueError):
            time.sleep(1)
    raise ValueError(f"项目尚未就绪；不要重复启动，请查看{log_path}。")


def stop_project(data_root, identity, current_project):
    if identity == current_project:
        raise ValueError("当前页面不能停止自身；请使用“停止项目助手.cmd”。")
    if identity not in projects(data_root):
        raise ValueError("项目不存在或源目录不在授权范围。")
    with _service_operations:
        root = Path(data_root) / "projects" / identity
        info_path = root / "local-service.json"
        if not info_path.is_file():
            raise ValueError("没有可核实的项目进程记录，未停止任何服务。")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        process = _recorded_process(info, identity)
        if process is None:
            return "该项目服务已停止；数据保留。"
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{info['port']}/project-health", timeout=2
            ) as response:
                health = json.load(response)
        except OSError:
            health = None
        if health and (health.get("app") != "rfq-kotaemon-project"
                       or health.get("project") != identity):
            raise ValueError("端口服务身份不符，未停止任何进程。")
        if health and health.get("business_busy"):
            raise ValueError("该项目正在录入、备份或查询，请等待操作结束后停止。")
        with closing(sqlite3.connect((root / "project.sqlite3").as_uri() + "?mode=ro", uri=True)) as conn:
            has_jobs = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='query_jobs'"
            ).fetchone()
            if has_jobs and conn.execute(
                "SELECT 1 FROM query_jobs WHERE status IN ('queued','running') LIMIT 1"
            ).fetchone():
                raise ValueError("该项目仍有查询任务；请先在对应页面停止任务并等待结束。")
        # Windows venv launchers can own a base-Python child with the same arguments.
        children = process.children(recursive=True)
        for child in reversed(children):
            verified = _recorded_process({"pid": child.pid, "port": info["port"]}, identity)
            if verified is not None:
                verified.terminate()
                verified.wait(timeout=15)
        process.terminate()
        process.wait(timeout=15)
        return "已停止所选项目服务；快照和已保存结果保留。"
