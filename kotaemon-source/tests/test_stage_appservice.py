import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "environment-setup"))
from prepare_nltk import extract_archive
from stage_appservice import acquire_file, copy_context_file, relative_path, stage_context
from build_appservice_code import (
    ROOT, _stage_headless_opencv, _stage_package, _write_requirements, build_package,
)


class AppServiceBuildContextTests(unittest.TestCase):
    def test_artifacts_must_match_size_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"fixture")
            correct = {"bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest()}
            acquire_file(root / "good", correct, local_file=source)
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                acquire_file(
                    root / "bad", {"bytes": 7, "sha256": "0" * 64}, local_file=source
                )
            with self.assertRaisesRegex(RuntimeError, "unexpected size"):
                acquire_file(
                    root / "short", {"bytes": 6, "sha256": correct["sha256"]},
                    local_file=source,
                )

    def test_unsafe_nltk_archive_is_not_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("punkt/../../outside", "fixture")
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive"):
                extract_archive(archive, root / "output", "punkt")
            self.assertFalse((root / "outside").exists())

    def test_context_never_overwrites_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty directory"):
                stage_context(directory)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_partial_local_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "all local"):
                stage_context(Path(directory) / "context", model_dir=directory)

    def test_windows_shell_lines_are_staged_with_linux_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "entrypoint.sh"
            target = Path(directory) / "staged.sh"
            source.write_bytes(b"#!/bin/sh\r\nexit 0\r\n")
            copy_context_file(source, target)
            self.assertEqual(target.read_bytes(), b"#!/bin/sh\nexit 0\n")

    def test_paths_must_stay_inside_context(self):
        for value in ("../secret", "/absolute", "a\\b", "C:/file"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_path(value)

    def test_code_deployment_requirements_do_not_override_the_locked_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            _write_requirements(package)
            constraints = {}
            for line in (ROOT / "environment-setup" / "upstream-constraints.txt").read_text(
                encoding="utf-8"
            ).splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(.+)", line.strip())
                if match:
                    constraints[match.group(1).lower().replace("_", "-")] = match.group(2)
            requested = []
            for line in (ROOT / "environment-setup" / "ocr-linux-requirements.txt").read_text(
                encoding="utf-8"
            ).splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(.+)", line.strip())
                if match:
                    requested.append(match.groups())
            for name, version in requested:
                locked = constraints.get(name.lower().replace("_", "-"))
                if locked:
                    with self.subTest(package=name):
                        self.assertEqual(version, locked)
            requirements = (package / "requirements.txt").read_text(encoding="utf-8")
            self.assertIn("-c upstream-constraints.txt", requirements)
            self.assertIn("rapidocr-onnxruntime==1.4.4", requirements)
            for name in ("kotaemon", "ktem"):
                with self.subTest(upstream=name):
                    self.assertIn(
                        f"{name} @ git+https://github.com/Cinnamon/kotaemon.git@"
                        "9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9"
                        f"#subdirectory=libs/{name}",
                        requirements,
                    )

    def test_code_package_keeps_embedding_manifest_and_linux_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            context = Path(directory) / "context"
            package = Path(directory) / "package"
            setup = root / "environment-setup"
            staged_setup = context / "environment-setup"
            setup.mkdir(parents=True)
            staged_setup.mkdir(parents=True)
            package.mkdir()
            (setup / "appservice-context-files.txt").write_text(
                "project_app.py\nenvironment-setup/embedding-download-manifest.json\n"
                "environment-setup/tiktoken-cache-manifest.json\n",
                encoding="utf-8",
            )
            (context / "project_app.py").write_text("pass\n", encoding="utf-8")
            (staged_setup / "upstream-constraints.txt").write_text("", encoding="utf-8")
            manifest = b'{"files": [], "local_directory": "fixture"}'
            (staged_setup / "embedding-download-manifest.json").write_bytes(manifest)
            for name in (
                "required-dependencies.in",
                "appservice-runtime-requirements.txt",
                "ocr-linux-requirements.txt",
                "ocr-model-manifest.json",
            ):
                (setup / name).write_text("", encoding="utf-8")
            (root / "start-appservice.sh").write_bytes(b"#!/bin/sh\r\nexit 0\r\n")

            assets = context / ".appservice-assets"
            (assets / "nltk" / "tokenizers" / "punkt").mkdir(parents=True)
            (assets / "nltk" / "tokenizers" / "punkt_tab" / "english").mkdir(parents=True)
            (assets / "tiktoken").mkdir(parents=True)
            tokenizers = []
            for name in ("cl100k_base.tiktoken", "o200k_base.tiktoken"):
                content = name.encode("ascii")
                (assets / "tiktoken" / name).write_bytes(content)
                tokenizers.append({
                    "name": name, "cache_key": name, "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            token_manifest = json.dumps({"files": tokenizers})
            (setup / "tiktoken-cache-manifest.json").write_text(token_manifest, encoding="utf-8")
            (staged_setup / "tiktoken-cache-manifest.json").write_text(
                token_manifest, encoding="utf-8"
            )
            icons = Path(directory) / "icons"
            icons.mkdir()
            for name in ("dark_mode", "expand", "rename", "delete", "new"):
                (icons / f"{name}.svg").write_text("<svg/>", encoding="utf-8")
            stopwords = Path(directory) / "stopwords"
            stopwords.mkdir()
            (stopwords / "english").write_text("fixture", encoding="utf-8")

            with patch("build_appservice_code.ROOT", root):
                _stage_package(context, package, icons, stopwords)
            self.assertEqual(
                (package / "environment-setup" / "embedding-download-manifest.json").read_bytes(),
                manifest,
            )
            self.assertEqual(
                (package / "environment-setup" / "tiktoken-cache-manifest.json")
                .read_text(encoding="utf-8"),
                token_manifest,
            )
            self.assertEqual(
                (package / "start-appservice.sh").read_bytes(), b"#!/bin/sh\nexit 0\n"
            )

    def test_headless_opencv_wheel_is_verified_and_safely_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup = root / "environment-setup"
            setup.mkdir()
            wheel = root / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("cv2/cv2.abi3.so", b"fixture")
                archive.writestr("cv2/LICENSE-3RD-PARTY.txt", b"notice")
                archive.writestr("opencv_python_headless.libs/libfixture.so", b"library")
            expected = {"filename": "fixture.whl", "bytes": wheel.stat().st_size,
                        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                        "url": "https://example.invalid/fixture.whl"}
            manifest = setup / "opencv-headless-manifest.json"
            manifest.write_text(json.dumps(expected), encoding="utf-8")
            with patch("build_appservice_code.ROOT", root):
                assets = root / "assets"
                _stage_headless_opencv(assets, wheel)
                self.assertEqual(
                    (assets / "ocr" / "opencv-headless" / "cv2" / "cv2.abi3.so").read_bytes(),
                    b"fixture",
                )
                self.assertTrue((assets / "ocr" / "opencv-headless-manifest.json").is_file())
                wheel.write_bytes(wheel.read_bytes() + b"changed")
                with self.assertRaises(RuntimeError):
                    _stage_headless_opencv(root / "bad-hash", wheel)

            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("../outside", b"unsafe")
            expected.update(bytes=wheel.stat().st_size,
                            sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
            manifest.write_text(json.dumps(expected), encoding="utf-8")
            with patch("build_appservice_code.ROOT", root):
                with self.assertRaises(ValueError):
                    _stage_headless_opencv(root / "bad-path", wheel)
            self.assertFalse((root / "outside").exists())

    def test_code_package_does_not_replace_a_racing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package.zip"
            original_link = os.link

            def stage_package(_context, package, _icons, _stopwords):
                (package / "fixture.txt").write_text("fixture", encoding="utf-8")

            def race(source, target):
                Path(target).write_bytes(b"existing")
                original_link(source, target)

            with patch("build_appservice_code.stage_context"), \
                    patch("build_appservice_code._stage_package", side_effect=stage_package), \
                    patch("build_appservice_code._stage_headless_opencv"), \
                    patch("build_appservice_code.os.link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    build_package(output, None, None, None, None, None)
            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).glob("*.partial")), [])
            output.unlink()
            with patch("build_appservice_code.stage_context"), \
                    patch("build_appservice_code._stage_package", side_effect=stage_package), \
                    patch("build_appservice_code._stage_headless_opencv"):
                build_package(output, None, None, None, None, None)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.read("fixture.txt"), b"fixture")


@unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "Requires Windows PowerShell")
class AppServiceCodeDeployTests(unittest.TestCase):
    def test_preflight_validates_real_cli_shapes_before_any_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake_az.py"
            fake.write_text(
                '''
import json
import os
import sys

args = sys.argv[1:]
state = json.loads(os.environ["FAKE_AZ_STATE"])
with open(os.environ["FAKE_AZ_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\\n")

if args[:2] == ["account", "show"]:
    print("fake-subscription")
    sys.exit(0)
if args[:2] == ["webapp", "show"]:
    data = {"id": "/subscriptions/fake-subscription/sites/fake", "kind": "app,linux",
            "httpsOnly": True, "appServicePlanId": "fake-plan",
            "defaultHostName": "fake.azurewebsites.net", "publicNetworkAccess": "Enabled"}
elif args[:3] == ["appservice", "plan", "show"]:
    if "--ids" not in args or args[args.index("--ids") + 1] != "fake-plan":
        sys.exit(2)
    data = {"properties": {"reserved": True, "numberOfWorkers": state["workers"]},
            "sku": {"capacity": state["capacity"], "name": "B2"}}
elif args[:3] == ["webapp", "config", "show"]:
    allowed = [{"action": "Allow", "ipAddress": state["allowed_ip"], "tag": "Default",
                "vnetSubnetResourceId": None}]
    if state.get("extra_allow"):
        allowed.append({"action": "Allow", "ipAddress": "0.0.0.0/0", "tag": "Default"})
    data = {"linuxFxVersion": "PYTHON|3.11",
            "ipSecurityRestrictionsDefaultAction": state["main_default"],
            "scmIpSecurityRestrictionsDefaultAction": "Deny",
            "scmIpSecurityRestrictionsUseMain": state["scm_uses_main"],
            "ipSecurityRestrictions": allowed}
elif args[0] == "rest":
    data = {"properties": {
        "platform": {"enabled": state["auth"], "configFilePath": None},
        "globalValidation": {"requireAuthentication": True,
                             "unauthenticatedClientAction": "Return401",
                             "excludedPaths": []},
        "identityProviders": {"azureActiveDirectory": {
            "enabled": True, "registration": {"clientId": "client",
                                               "openIdIssuer": "https://issuer.example"}}}}}
else:
    print("{}")
    sys.exit(0)

query = args[args.index("--query") + 1].strip("{}")
result = {}
for entry in query.split(","):
    alias, path = entry.split(":")
    value = data
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    result[alias] = value
print(json.dumps(result))
'''.lstrip(),
                encoding="utf-8",
            )
            cli = root / "az.cmd"
            cli.write_text(f'@echo off\n"{sys.executable}" "{fake}" %*\nexit /b %errorlevel%\n',
                           encoding="utf-8")
            package = root / "package.zip"
            package.write_bytes(b"fixture")
            log = root / "calls.jsonl"
            script = ROOT / "environment-setup" / "deploy-appservice-code.ps1"

            for case in (
                {"omit_subscription": True, "expected": "SubscriptionId"},
                {"subscription": "other-subscription", "expected": "different subscription"},
                {"invalid_cli": True, "expected": "supplied -AzureCli path"},
                {"auth": False, "expected": "Configure and verify required Entra authentication"},
                {"workers": 2, "expected": "exactly one worker"},
                {"capacity": 2, "expected": "exactly one worker"},
                {"expected": "Source package upload accepted", "success": True},
                {"cli_on_path": True, "expected": "Source package upload accepted", "success": True},
                {"auth": False, "ip_only": True, "expected": "Source package upload accepted",
                 "success": True},
                {"auth": False, "ip_only": True, "runtime_root": "/home/rfq-smoke",
                 "expected": "Source package upload accepted", "success": True},
                {"auth": False, "ip_only": True, "runtime_root": "/home/../outside",
                 "expected": "safe absolute directory under /home"},
                {"auth": False, "ip_only": True, "cidr": "0.0.0.0/0",
                 "expected": "one canonical IPv4 /32"},
                {"auth": False, "ip_only": True, "extra_allow": True,
                 "expected": "allow only the confirmed company egress IP"},
                {"auth": False, "ip_only": True, "main_default": "Allow",
                 "expected": "deny-by-default"},
                {"auth": False, "ip_only": True, "scm_uses_main": False,
                 "expected": "deny-by-default"},
                {"ip_only": True, "expected": "authentication must be reviewed"},
            ):
                with self.subTest(case=case):
                    log.unlink(missing_ok=True)
                    env = os.environ.copy()
                    env["FAKE_AZ_LOG"] = str(log)
                    state = {"workers": 1, "capacity": 1, "auth": True,
                             "allowed_ip": "203.0.113.10/32", "main_default": "Deny",
                             "scm_uses_main": True}
                    state.update(case)
                    env["FAKE_AZ_STATE"] = json.dumps(state)
                    if case.get("cli_on_path"):
                        env["PATH"] = str(root) + os.pathsep + env["PATH"]
                    arguments = [
                        shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-File", str(script),
                        "-ResourceGroup", "fake", "-WebAppName", "fake",
                        "-PackagePath", str(package),
                    ]
                    if not case.get("cli_on_path"):
                        arguments.extend([
                            "-AzureCli", str(root / "missing.cmd" if case.get("invalid_cli") else cli),
                        ])
                    if not case.get("omit_subscription"):
                        arguments.extend([
                            "-SubscriptionId", case.get("subscription", "fake-subscription"),
                        ])
                    if case.get("ip_only"):
                        arguments.extend([
                            "-CompanyIpOnly", "-CompanyEgressCidr",
                            case.get("cidr", "203.0.113.10/32"),
                        ])
                    if case.get("runtime_root"):
                        arguments.extend(["-RuntimeRoot", case["runtime_root"]])
                    completed = subprocess.run(
                        arguments,
                        capture_output=True, text=True, errors="replace", env=env, timeout=30,
                    )
                    self.assertIn(case["expected"], completed.stdout + completed.stderr)
                    calls = ([json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                             if log.exists() else [])
                    deployments = [args for args in calls if args[:2] == ["webapp", "deploy"]]
                    if case.get("success"):
                        self.assertEqual(completed.returncode, 0)
                        self.assertEqual(len(deployments), 1)
                        self.assertEqual(
                            deployments[0][deployments[0].index("--async") + 1], "true"
                        )
                        self.assertEqual(
                            deployments[0][deployments[0].index("--track-status") + 1], "false"
                        )
                        config = next(args for args in calls if args[:3] == ["webapp", "config", "set"])
                        self.assertNotIn("--linux-fx-version", config)
                        settings = next(args for args in calls if args[:4] ==
                                        ["webapp", "config", "appsettings", "set"])
                        mode = "company-ip" if case.get("ip_only") else "entra"
                        self.assertIn(f"RFQ_APP_SERVICE_ACCESS_MODE={mode}", settings)
                        runtime_root = case.get("runtime_root", "/home/rfq")
                        self.assertIn(f"RFQ_RUNTIME_HOME={runtime_root}/data", settings)
                        self.assertIn(f"RFQ_AUTHORIZED_SOURCE={runtime_root}/source", settings)
                    else:
                        self.assertNotEqual(completed.returncode, 0)
                        self.assertFalse(deployments)
                        self.assertFalse(any(args[:3] == ["webapp", "config", "set"]
                                             or args[:4] == ["webapp", "config", "appsettings", "set"]
                                             for args in calls))


if __name__ == "__main__":
    unittest.main()
