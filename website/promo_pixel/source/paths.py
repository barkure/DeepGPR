"""Repository-relative paths shared by the pixel promo scripts."""
import os
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[2]
WEBSITE_DIR = REPO_ROOT / "website"
# Keep footage data beside its source by default, regardless of the caller's
# working directory. A separate workspace can be selected for regeneration.
DATA_DIR = Path(os.environ.get("DEEPGPR_PROMO_DATA", SOURCE_DIR)).expanduser().resolve()
