from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from file_lock import exclusive_file_lock


class FileLockTests(unittest.TestCase):
    def test_lock_excludes_another_process_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "service.lock"
            script = (
                "import sys\n"
                "from file_lock import FileLockError, exclusive_file_lock\n"
                "try:\n"
                "    with exclusive_file_lock(sys.argv[1]):\n"
                "        raise SystemExit(0)\n"
                "except FileLockError:\n"
                "    raise SystemExit(23)\n"
            )
            with exclusive_file_lock(lock_path):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(lock_path)],
                    cwd=Path(__file__).resolve().parent.parent,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 23, result.stderr)
            with exclusive_file_lock(lock_path):
                pass


if __name__ == "__main__":
    unittest.main()
