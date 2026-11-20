"""Settings exclusively for the read-only Kotaemon layout preview."""

from theflow.settings.default import *  # noqa: F403
from runtime_paths import preview_data_path

KH_DATABASE = "sqlite://"
KH_ENABLE_ALEMBIC = False
KH_DEMO_MODE = False
KH_SSO_ENABLED = True
KH_FEATURE_USER_MANAGEMENT = False
KH_FEATURE_CHAT_SUGGESTION = False
KH_WEB_SEARCH_BACKEND = None
KH_LLMS = {}
KH_EMBEDDINGS = {}
KH_RERANKINGS = {}
KH_REASONINGS = []
KH_INDICES = []
KH_INDEX_TYPES = []
KH_APP_DATA_DIR = preview_data_path()
KH_USER_DATA_DIR = KH_APP_DATA_DIR / "user_data"
KH_MARKDOWN_OUTPUT_DIR = KH_APP_DATA_DIR / "markdown"
KH_CHUNKS_OUTPUT_DIR = KH_APP_DATA_DIR / "chunks"
KH_ZIP_OUTPUT_DIR = KH_APP_DATA_DIR / "zip_output"
KH_ZIP_INPUT_DIR = KH_APP_DATA_DIR / "zip_input"
KH_FILESTORAGE_PATH = str(KH_USER_DATA_DIR / "files")
for directory in (
    KH_USER_DATA_DIR, KH_MARKDOWN_OUTPUT_DIR, KH_CHUNKS_OUTPUT_DIR,
    KH_ZIP_OUTPUT_DIR, KH_ZIP_INPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
