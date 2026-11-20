from pathlib import Path
from unittest.mock import patch
import unittest

import project_registry
from project_registry import SOURCE_ENV, authorized_source


class AuthorizedSourceTests(unittest.TestCase):
    def test_default_is_sibling_review_directory(self):
        with patch.dict("os.environ", {SOURCE_ENV: ""}):
            expected = Path(project_registry.__file__).resolve().parent.parent / "评审后"
            self.assertEqual(authorized_source(), expected)

    def test_environment_override(self):
        with patch.dict("os.environ", {SOURCE_ENV: "/data/approved"}):
            self.assertEqual(authorized_source(), Path("/data/approved"))


if __name__ == "__main__":
    unittest.main()
