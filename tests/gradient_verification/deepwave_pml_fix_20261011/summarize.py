"""Rebuild the PML-fix result tables and standalone scientific figures."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepgpr-audit-matplotlib")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FILES = ("basic_cpu", "basic_cuda", "three_d_cpu", "three_d_cuda",
         "time_cpu", "time_cuda", "pml_cpu", "pml_cuda")
all_rows, tables = [], []
for name in FILES:
    payload = json.loads((DATA / (name + ".json")).read_text())
    rows = payload["results"]
    all_rows.extend(rows)
    tables.append(dict(file=name + ".json", count=len(rows),
                       verdicts=dict(Counter(r["verdict"] for r in rows)),
                       hashes=payload["environment"]["hashes"]))
assert len(all_rows) == 573
assert not any(r["verdict"] == "FAIL" for r in all_rows)
manifest = json.loads((HERE / "native_build_manifest.json").read_text())
for table in tables:
    for library in ("deepgpr.so", "deepgpr_cpu.so"):
        assert table["hashes"]["src/DeepGPR/lib/" + library] == manifest["libraries"][library]["sha256"]
summary = dict(primary_direction_count=len(all_rows),
               verdicts=dict(Counter(r["verdict"] for r in all_rows)), tables=tables,
               policy="No FD boundary mask; fixed profiles; original 0.2% multistep/Taylor/noise rules",
               source_hashes=manifest["source_sha256"],
               numerical_uncertainty="Noise-qualified PASS is distinguished from strict PASS; inconclusive tests are not passes.")
(DATA / "primary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (DATA / "primary_directions.csv").open("w", newline="") as handle:
    columns = ("label", "path", "kind", "direction", "nt", "dt", "order", "g_ad", "g_fd",
               "rel_err", "pair_rel_err", "fit_slope", "noise", "verdict", "reason")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)
figures = HERE / "figures"
figures.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.2, 4.2))
labels = ("Pass", "Noise-qualified pass", "Inconclusive", "Fail")
before, after = (234, 0, 150, 189), (450, 15, 108, 0)
x = np.arange(4)
ax.bar(x - .18, before, .36, label="Previous public profiles", color="#8995a5")
ax.bar(x + .18, after, .36, label="Fixed profiles + PML sensitivities", color="#167d8d")
for offset, values in ((-.18, before), (.18, after)):
    for position, value in zip(x + offset, values):
        ax.text(position, value + 5, str(value), ha="center", va="bottom")
ax.set_xticks(x, labels)
ax.set_ylabel("Directional tests")
ax.set_ylim(0, 520)
ax.set_title("Matched 573-test physical matrix (0.2% rules unchanged)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(figures / "primary_before_after.png", dpi=180)
fig.savefig(figures / "primary_before_after.svg")
plt.close(fig)
comparison = json.loads((DATA / "native_before_after_all.json").read_text())
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for probe in comparison["results"]:
    if probe["path"] not in ("cpu", "cuda_general"):
        continue
    row = next(r for r in probe["cases"] if r["parameter"] == "eps")
    steps = [r["h"] for r in row["rows"]]
    before_errors = [abs(r["fd"] - row["before_ad"]) / max(abs(r["fd"]), abs(row["before_ad"]), 1.e-30)
                     for r in row["rows"]]
    after_errors = [r["fd_rel_err"] for r in row["rows"]]
    label = f"{probe['dims']}D {probe['path']} edge"
    ax.loglog(steps, before_errors, "--", alpha=.55, label=label + " before")
    ax.loglog(steps, after_errors, "o-", markersize=3, label=label + " after")
ax.axhline(.002, color="black", linewidth=.8, linestyle=":", label="0.2%")
ax.set_xlabel("Relative perturbation step h")
ax.set_ylabel("Directional derivative relative error")
ax.set_title("Identical forward profiles: native PML sensitivity correction")
ax.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(figures / "fixed_profile_edge_fd.png", dpi=180)
fig.savefig(figures / "fixed_profile_edge_fd.svg")
plt.close(fig)
artifacts = {}
for path in (DATA / "primary_summary.json", DATA / "primary_directions.csv",
             HERE / "physical_verification.py", HERE / "native_before_after.py", HERE / "long_time_probe.py"):
    artifacts[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
(HERE / "evidence_manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
print(json.dumps(summary["verdicts"]))
