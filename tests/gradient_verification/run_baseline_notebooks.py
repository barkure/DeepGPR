"""Run the existing gradient notebooks without touching tests/results.

The notebooks save their JSON reports through verification_utils.save_report,
which writes to verification_utils.RESULTS_DIR; this wrapper points that
directory at tests/gradient_verification/data/baseline_notebooks first.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS))
sys.dont_write_bytecode = True

import run_notebook  # noqa: E402
import verification_utils as vu  # noqa: E402

OUT = HERE / "data" / "baseline_notebooks"


def digest(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).glob("*")) if p.is_file()}


if __name__ == "__main__":
    vu.RESULTS_DIR = OUT
    before = digest(TESTS / "results")
    for name in sys.argv[1:]:
        run_notebook.execute_notebook((TESTS / name).resolve())
    after = digest(TESTS / "results")
    print("tests/results unchanged:", before == after)
    if before != after:
        raise SystemExit("tests/results was modified")
