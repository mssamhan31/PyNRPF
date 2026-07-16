"""Build the readable journal experiment notebooks from reviewed source cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def write_notebook(path: Path, cells: list) -> Path:
    notebook = nbf.v4.new_notebook(cells=cells, metadata=KERNEL_METADATA)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


def build_02a(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            r"""
# 02a m9_pbm Method And Worked Example

**Research question.** How does `m9_pbm` turn an N-shaped reverse-power-flow
sign error into a physically interpretable candidate correction?

This notebook explains the method before any model comparison is performed. It
uses one fixed, previously chosen example: **Alpha substation F on 17 February
2024**. The notebook reads the final Alpha Parquet file, checks its hash and
schema, reconstructs the two possible underlying demand curves from the
equations below, and writes one two-panel publication figure.

**Inputs:** final Alpha data and the versioned experiment configuration.  
**Outputs:** one plot-data CSV, one 300-dpi PNG, and one reproducibility manifest.  
**Expected runtime:** under one minute on a laptop.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Visible Configuration

The path search below starts from the current working directory, so this
notebook can run on another laptop after cloning the repository. The displayed
configuration is the source of truth for the candidate bounds and feature
constants used throughout Notebooks 02a-02g.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_dataset,
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    validate_input_hashes,
    write_csv,
    write_manifest,
)
from _m9_pbm_features import (  # noqa: E402
    CandidateSpec,
    bridge_line,
    reconstruct_demand,
)
from _m9_pbm_plotting import plot_method_example  # noqa: E402

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SPEC = CandidateSpec.from_config(CONFIG)
SLUG = "02a_m9_pbm_method_example"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)

display(pd.Series(CONFIG["m9_pbm"]["example"], name="worked_example"))
display(pd.Series(CONFIG["m9_pbm"]["candidate_windows"], name="candidate_windows"))
"""
        ),
        markdown(
            """
## 2. Input Validation

The final dataset hash must match the configuration before results are written.
For the worked example we also require exactly 96 quarter-hour readings, one
positive day label, and 16 positive interval labels. These checks prevent a
silent label refresh or an incomplete day from changing the figure.
"""
        ),
        code(
            """
hash_audit = validate_input_hashes(PATHS, CONFIG)
display(hash_audit)

alpha = load_dataset("alpha", article_root=ARTICLE_ROOT, config=CONFIG)
example_cfg = CONFIG["m9_pbm"]["example"]
example = alpha.loc[
    alpha["substation_id"].eq(example_cfg["substation_id"])
    & alpha["date"].eq(example_cfg["date"])
].copy()
example = example.sort_values("timestamp").reset_index(drop=True)
example["slot"] = example["timestamp"].dt.hour * 4 + example["timestamp"].dt.minute // 15

assert len(example) == 96, f"Expected 96 readings, found {len(example)}."
assert example["slot"].nunique() == 96, "The example does not contain 96 unique slots."
assert bool(example["label_day"].max())
assert int(example["label_interval"].sum()) == 16

input_audit = pd.DataFrame(
    [{
        "substation_id": example_cfg["substation_id"],
        "date": example_cfg["date"],
        "readings": len(example),
        "labelled_rpf_intervals": int(example["label_interval"].sum()),
        "first_labelled_time": example.loc[example["label_interval"], "timestamp"].min(),
        "last_labelled_time": example.loc[example["label_interval"], "timestamp"].max(),
    }]
)
display(input_audit)
"""
        ),
        markdown(
            r"""
## 3. Physical Interpretation And Core Equations

During a true reverse-power-flow interval, net load should be negative because
solar export exceeds local demand. A polarity or sign error instead makes this
export appear as a positive midday bump, often with an N-shaped rise and fall.
`m9_pbm` asks which of two interpretations produces the more physically
plausible full-day underlying demand curve.

The **no-correction interpretation** is

$$
U_{no}(t) = S(t) + y(t).
$$

For a candidate window $W$, the **candidate-corrected interpretation** is

$$
U_{corr,W}(t) =
\begin{cases}
S(t)-y(t), & t \in W,\\
S(t)+y(t), & t \notin W.
\end{cases}
$$

If the day is classified as positive, its corrected net-load output is

$$
y_{corr,W}(t) =
\begin{cases}
-y(t), & t \in W,\\
y(t), & t \notin W.
\end{cases}
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $t$ | One 15-minute timestamp in a substation-day. |
| $y(t)$ | Observed net load in MW. A sign error appears as an incorrectly positive RPF bump. |
| $S(t)$ | Estimated solar generation in MW. |
| $W$ | One contiguous candidate correction window. |
| $U_{no}(t)$ | Reconstructed underlying demand if no sign correction is applied. |
| $U_{corr,W}(t)$ | Reconstructed demand if the sign is corrected only inside $W$. |
| $y_{corr,W}(t)$ | Net load written after a positive day decision. |
"""
        ),
        markdown(
            r"""
## 4. Candidate Windows, Features, And Decisions

Candidate generation is deterministic and uses no labels. It scans the daytime
range from 06:00 through 18:00. A window must last from 2 to 32 slots (30
minutes to 8 hours), and its midpoint must be within 14 slots (3.5 hours) of
that day's solar peak. Each candidate is evaluated on the full-day
reconstructions, with local calculations focused on $W$ and its shoulders
$\Omega(W)$.

For active feature set $A$, candidate selection and day classification are
separate operations:

$$
Score(W)=\sum_{i\in A}w_iF_i(W),
\qquad
W_d^*=\operatorname*{arg\,max}_{W} Score(W),
$$

$$
\widehat{RPF}(d)=\mathbb{1}\{Score(W_d^*)\geq\tau\}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $d$ | One substation-day. |
| $A$ | The active set of physical features. |
| $F_i(W)$ | Value of feature $i$ for candidate $W$. |
| $w_i$ | Nonnegative weight assigned to feature $i$. |
| $Score(W)$ | Weighted physical plausibility score for candidate $W$. |
| $W_d^*$ | Highest-scoring candidate on day $d$; ties are resolved deterministically. |
| $\tau$ | Day threshold learned from training substations only. |
| $\mathbb{1}\{\cdot\}$ | Indicator equal to one when its condition is true. |

Only when $\widehat{RPF}(d)=1$ does $W_d^*$ become the predicted correction
interval. A negative day decision leaves every observed reading unchanged.
"""
        ),
        markdown(
            r"""
## 5. The Nine Physical Feature Concepts

The features describe complementary physical evidence. Positive improvement
features mean that the candidate correction makes demand more plausible.

### F1: bridge improvement

Let $L_W(t)$ be the straight line joining demand immediately before and after
$W$:

$$
E_{bridge}(U,W)=\operatorname{median}_{t\in W}|U(t)-L_W(t)|,
$$

$$
F_1(W)=\frac{E_{bridge}(U_{no},W)-E_{bridge}(U_{corr,W},W)}
{E_{bridge}(U_{no},W)+E_{bridge}(U_{corr,W},W)+\epsilon}.
$$

Here $L_W(t)$ is the linear bridge, $E_{bridge}$ is median absolute bridge
error, and $\epsilon$ is a small positive constant that prevents division by
zero.

### F2: roughness improvement

$$
TV(U,\Omega(W))=\sum_{t,t+\Delta t\in\Omega(W)}|U(t+\Delta t)-U(t)|,
$$

$$
F_2(W)=\frac{TV(U_{no},\Omega(W))-TV(U_{corr,W},\Omega(W))}
{TV(U_{no},\Omega(W))+TV(U_{corr,W},\Omega(W))+\epsilon}.
$$

$\Omega(W)$ is $W$ plus three-slot shoulders, $\Delta t$ is 15 minutes, and
$TV$ is total variation over adjacent readings.

### F3: slope-continuity improvement

$$
J(U,W)=|m_{out,left}(U)-m_{in,left}(U)|
+|m_{in,right}(U)-m_{out,right}(U)|,
$$

$$
F_3(W)=\frac{J(U_{no},W)-J(U_{corr,W},W)}
{J(U_{no},W)+J(U_{corr,W},W)+\epsilon}.
$$

$m_{out,left}$ and $m_{out,right}$ are robust slopes outside the boundaries;
$m_{in,left}$ and $m_{in,right}$ are slopes just inside them; $J$ is their
summed mismatch.

### F4-F7: duration, N-height, solar strength, and peak alignment

$$
F_4(W)=\operatorname{clip}\left(\frac{duration_{hours}(W)}{1.5},0,1\right),
$$

$$
F_5(W)=\operatorname{clip}\left(
\frac{\max_{t\in W}y(t)-\max(y(t_{left}),y(t_{right}))}{P_{scale}},0,1\right),
$$

$$
F_6(W)=\operatorname{clip}\left(\frac{P95_{t\in W}S(t)}{S_{substation}},0,1\right),
$$

$$
F_7(W)=\operatorname{clip}\left(
1-\frac{|midpoint(W)-t_{solar\ peak}|}{3.5\ hours},0,1\right).
$$

$duration_{hours}$ is window length; $t_{left}$ and $t_{right}$ are its boundary
slots; $P_{scale}$ is a robust day power scale; $P95$ is the 95th percentile;
$S_{substation}$ is the substation's median historical daytime-solar P95; and
$t_{solar\ peak}$ is that day's maximum-solar slot.

### F8-F9: substation-relative core score

$$
core(W)=F_1(W)+F_2(W)+F_3(W),
$$

$$
F_8(W)=robust\_bound\{core(W)-median(core_{best,day})\},
$$

$$
F_9(d)=2\,percentile\_rank(core_{best,d})-1.
$$

$core_{best,d}$ is the highest core score on day $d$. The median and percentile
rank are calculated within the same substation using unlabelled scores only.
F8 measures centred magnitude; F9 measures relative daily rank. Neither uses a
manual RPF label.
"""
        ),
        markdown(
            """
## 6. Reconstruct The Worked Example

The labelled Alpha interval is used here only to define the worked window. It
runs from slot 40 through slot 55, or 10:00 through 13:45 inclusive. The bridge
anchors are the immediately adjacent readings at 09:45 and 14:00. Later model
notebooks generate and select candidates without reading these labels.
"""
        ),
        code(
            """
net_load = example["net_load_MW"].to_numpy(dtype=float)
solar = example["solar_MW"].to_numpy(dtype=float)
true_slots = example.loc[example["label_interval"], "slot"].to_numpy(dtype=int)
left_slot, right_slot = int(true_slots.min()), int(true_slots.max())
assert (left_slot, right_slot) == (40, 55)

uncorrected, corrected = reconstruct_demand(net_load, solar, left_slot, right_slot)
bridge_inside, (left_anchor, right_anchor) = bridge_line(
    uncorrected, left_slot, right_slot, SPEC
)
bridge_slots = np.arange(left_anchor, right_anchor + 1)
bridge_values = np.interp(
    bridge_slots,
    [left_anchor, right_anchor],
    [uncorrected[left_anchor], uncorrected[right_anchor]],
)

plot_data = pd.DataFrame(
    {
        "substation_id": example["substation_id"],
        "date": example["date"],
        "timestamp": example["timestamp"],
        "slot": example["slot"],
        "observed_net_load_MW": net_load,
        "solar_generation_MW": solar,
        "uncorrected_demand_MW": uncorrected,
        "corrected_demand_MW": corrected,
        "true_interval": example["label_interval"],
        "candidate_window": example["slot"].between(left_slot, right_slot),
        "bridge_anchor": example["slot"].isin([left_anchor, right_anchor]),
        "linear_bridge_MW": np.nan,
    }
)
plot_data.loc[plot_data["slot"].isin(bridge_slots), "linear_bridge_MW"] = bridge_values

assert np.allclose(
    plot_data.loc[plot_data["candidate_window"], "linear_bridge_MW"],
    bridge_inside,
)
display(
    pd.Series(
        {
            "candidate_start": plot_data.loc[plot_data["candidate_window"], "timestamp"].min(),
            "candidate_end": plot_data.loc[plot_data["candidate_window"], "timestamp"].max(),
            "left_anchor": plot_data.loc[plot_data["slot"].eq(left_anchor), "timestamp"].iloc[0],
            "right_anchor": plot_data.loc[plot_data["slot"].eq(right_anchor), "timestamp"].iloc[0],
        },
        name="worked_window",
    )
)
"""
        ),
        markdown(
            """
## 7. Publication Figure And Source Data

Panel (a) retains the measurements the method actually sees. Panel (b) changes
only the demand interpretation inside the candidate window. The bridge is
anchored outside that window, so it provides a local reference without
flattening or replacing the reconstructed demand curve.
"""
        ),
        code(
            """
TABLE_PATH = OUTPUT_DIRS["tables"] / "table01_alpha_F_2024-02-17_plot_data.csv"
FIGURE_PATH = OUTPUT_DIRS["figures"] / "fig01_m9_pbm_alpha_F_2024-02-17.png"

write_csv(plot_data, TABLE_PATH)
plot_method_example(plot_data, FIGURE_PATH)

display(plot_data.head())
display(FIGURE_PATH)
"""
        ),
        markdown(
            """
## 8. Findings And Limitations

For this worked day, the positive midday net-load bump creates an implausible
feature in the no-correction demand curve. Applying the candidate sign change
from 10:00 through 13:45 yields a smoother physical interpretation relative to
the two outside anchors. This figure demonstrates the mechanism; it is not a
performance estimate.

The example window comes from the Alpha reference label and must not be confused
with a fitted model prediction. Candidate selection, substation-held-out
threshold selection, and interval evaluation are performed in later notebooks.
"""
        ),
        markdown(
            """
## 9. Reproducibility Manifest And Output Inventory

The manifest records current input hashes, configuration hash, environment,
elapsed time, row counts, and output hashes. Only declared final inputs are read.
"""
        ),
        code(
            """
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[PATHS.config, PATHS.final_data / "dataset_alpha.parquet"],
    outputs=[TABLE_PATH, FIGURE_PATH],
    row_counts={
        "example_readings": len(plot_data),
        "labelled_intervals": int(plot_data["true_interval"].sum()),
    },
)
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)

output_inventory = pd.DataFrame(
    {
        "type": ["table", "figure", "manifest"],
        "path": [TABLE_PATH, FIGURE_PATH, MANIFEST_PATH],
    }
)
output_inventory["exists"] = output_inventory["path"].map(Path.exists)
output_inventory["bytes"] = output_inventory["path"].map(lambda path: path.stat().st_size)
display(output_inventory)
assert output_inventory["exists"].all()
assert (output_inventory["bytes"] > 0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02a_m9_pbm_method_and_example.ipynb", cells)


def build_02b(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02b m9_pbm Candidate Windows And Physical Features

**Research question.** Can every valid Alpha and Beta candidate window be
represented once, reproducibly, before model variants select different windows?

This notebook is the only expensive physical-feature stage. It generates every
label-free 30-minute to 8-hour candidate, computes F1-F9, writes resumable
substation partitions, and consolidates them into the cache used by Notebooks
02c-02g. It deliberately does **not** reduce each day to the candidate preferred
by an older model.

**Inputs:** final Alpha and Beta data plus the experiment configuration.  
**Outputs:** a local candidate cache, a day-input cache, three compact audit
tables, and a reproducibility manifest.  
**Expected runtime:** approximately 25-40 minutes for a clean run; under one
minute when all validated substation partitions already exist.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Configuration

The displayed values define the candidate universe. Labels and reviewer
confidence are not passed to candidate generation or F1-F9 calculation. They
remain only in the one-row-per-day audit used to prove a complete later join.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    consolidate_parquet_files,
    load_dataset,
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    to_day_arrays,
    validate_input_hashes,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import (  # noqa: E402
    FEATURE_COLUMNS,
    CandidateSpec,
    build_substation_candidate_features,
    compute_candidate_features,
    substation_solar_scale,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SPEC = CandidateSpec.from_config(CONFIG)
SLUG = "02b_m9_pbm_candidate_features"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
PARTITION_DIR = OUTPUT_DIRS["intermediate"] / "_partitions"
PARTITION_DIR.mkdir(parents=True, exist_ok=True)

display(pd.Series(CONFIG["m9_pbm"]["candidate_windows"], name="candidate_windows"))
display(pd.Series(CONFIG["m9_pbm"]["features"], name="features"))
"""
        ),
        markdown(
            """
## 2. Preflight Validation And Runtime Benchmark

All final-data hashes are checked before an existing partition can be reused.
The benchmark computes real Alpha candidates and projects the clean-run feature
time. It fails before the full scan if the projection exceeds one hour, making
an unexpectedly slow environment visible rather than starting an uncontrolled
job.
"""
        ),
        code(
            """
hash_audit = validate_input_hashes(PATHS, CONFIG)
display(hash_audit)

alpha_probe = load_dataset("alpha", article_root=ARTICLE_ROOT, config=CONFIG)
alpha_probe = alpha_probe.loc[alpha_probe["substation_id"].eq("alpha_F")]
probe_days = to_day_arrays(alpha_probe)
probe_scale = substation_solar_scale(probe_days.solar[:20], SPEC)

benchmark_started = time.perf_counter()
benchmark_candidates = 0
for day_index in range(20):
    candidate_probe, _ = compute_candidate_features(
        probe_days.net_load[day_index],
        probe_days.solar[day_index],
        substation_solar_scale=probe_scale,
        spec=SPEC,
    )
    benchmark_candidates += len(candidate_probe)
benchmark_seconds = time.perf_counter() - benchmark_started
expected_days = sum(CONFIG["datasets"]["expected_substation_days"].values())
projected_minutes = benchmark_seconds / 20 * expected_days / 60
benchmark = pd.DataFrame(
    [{
        "benchmark_days": 20,
        "mean_candidates_per_day": benchmark_candidates / 20,
        "seconds_per_day": benchmark_seconds / 20,
        "projected_clean_feature_minutes": projected_minutes,
    }]
)
display(benchmark)
assert projected_minutes < 60, (
    f"Projected feature runtime is {projected_minutes:.1f} minutes; optimise or "
    "inspect the environment before launching the full scan."
)
del alpha_probe, probe_days, candidate_probe
"""
        ),
        markdown(
            """
## 3. Cache Design And Leakage Boundary

Each candidate row contains only its substation-day key, window geometry, and
physical features. The candidate file contains no day label, interval label, or
reviewer confidence. A separate day-input cache retains those fields solely for
auditing and later evaluation joins.

F1-F7 are computed directly from observed net load and solar generation. F8 and
F9 use unlabelled within-substation distributions of the daily best
F1+F2+F3 core score. One partition is written per substation, so an interrupted
run can resume without recomputing completed substations. The final consolidated
Parquet file is streamed from those partitions instead of concatenating the
whole candidate universe in memory.
"""
        ),
        code(
            """
partition_records = []
for dataset in ["alpha", "beta"]:
    dataset_frame = load_dataset(dataset, article_root=ARTICLE_ROOT, config=CONFIG)
    substations = sorted(dataset_frame["substation_id"].unique())
    for substation in substations:
        stem = f"{dataset}_{substation}"
        candidate_path = PARTITION_DIR / f"{stem}_candidates.parquet"
        audit_path = PARTITION_DIR / f"{stem}_day_audit.parquet"
        daily_path = PARTITION_DIR / f"{stem}_daily_best.parquet"
        reusable = (
            CONFIG["execution"]["resume_validated_intermediates"]
            and candidate_path.exists()
            and audit_path.exists()
            and daily_path.exists()
        )

        if not reusable:
            substation_frame = dataset_frame.loc[
                dataset_frame["substation_id"].eq(substation)
            ].copy()
            day_arrays = to_day_arrays(substation_frame, SPEC.slots_per_day)
            candidates, audit, daily_best = build_substation_candidate_features(
                dataset=dataset,
                keys=day_arrays.keys,
                net_load_days=day_arrays.net_load,
                solar_days=day_arrays.solar,
                spec=SPEC,
            )
            write_parquet(candidates, candidate_path)
            write_parquet(audit, audit_path)
            write_parquet(daily_best, daily_path)
            del substation_frame, day_arrays, candidates, audit, daily_best

        candidate_metadata = pq.read_metadata(candidate_path)
        audit_metadata = pq.read_metadata(audit_path)
        partition_records.append(
            {
                "dataset": dataset,
                "substation_id": substation,
                "candidate_path": candidate_path,
                "audit_path": audit_path,
                "daily_path": daily_path,
                "candidate_rows": candidate_metadata.num_rows,
                "substation_days": audit_metadata.num_rows,
                "reused": reusable,
            }
        )
        print(
            f"{dataset} {substation}: {candidate_metadata.num_rows:,} candidates "
            f"across {audit_metadata.num_rows:,} days ({'reused' if reusable else 'built'})",
            flush=True,
        )
    del dataset_frame

partition_index = pd.DataFrame(partition_records)
display(partition_index)
"""
        ),
        markdown(
            """
## 4. Consolidate Local Intermediates

The candidate cache is intentionally local-only because it is large but fully
reproducible. Compact tables, notebooks, configuration, figures, and manifests
remain suitable for Git. The day-input cache is also local-only and contains one
row per Alpha or Beta substation-day.
"""
        ),
        code(
            """
CANDIDATE_CACHE = OUTPUT_DIRS["intermediate"] / "candidate_feature_cache.parquet"
DAY_INPUT_CACHE = OUTPUT_DIRS["intermediate"] / "day_input_cache.parquet"

candidate_parts = partition_index["candidate_path"].tolist()
audit_parts = partition_index["audit_path"].tolist()
consolidate_parquet_files(candidate_parts, CANDIDATE_CACHE)
consolidate_parquet_files(audit_parts, DAY_INPUT_CACHE)

candidate_metadata = pq.read_metadata(CANDIDATE_CACHE)
day_metadata = pq.read_metadata(DAY_INPUT_CACHE)
assert candidate_metadata.num_rows == int(partition_index["candidate_rows"].sum())
assert day_metadata.num_rows == int(partition_index["substation_days"].sum())
assert day_metadata.num_rows == sum(CONFIG["datasets"]["expected_substation_days"].values())

display(
    pd.Series(
        {
            "candidate_rows": candidate_metadata.num_rows,
            "substation_days": day_metadata.num_rows,
            "candidate_cache_GiB": CANDIDATE_CACHE.stat().st_size / 1024**3,
            "day_cache_MiB": DAY_INPUT_CACHE.stat().st_size / 1024**2,
        },
        name="cache",
    )
)
"""
        ),
        markdown(
            """
## 5. Candidate Counts, Feature Quality, And Label-Join Audit

The first table checks candidate coverage and input gaps by dataset and
substation. The second verifies that every physical feature is finite and
records its observed range. The third proves that each final Alpha/Beta
substation-day has exactly one audit row available for later label joins.
"""
        ),
        code(
            """
audit_frame = pd.concat(
    [pd.read_parquet(path) for path in partition_index["audit_path"]],
    ignore_index=True,
)
candidate_counts = (
    audit_frame.groupby(["dataset", "substation_id"], as_index=False)
    .agg(
        substation_days=("date", "size"),
        total_candidates=("candidate_count", "sum"),
        minimum_candidates_per_day=("candidate_count", "min"),
        mean_candidates_per_day=("candidate_count", "mean"),
        maximum_candidates_per_day=("candidate_count", "max"),
        input_missing_net_slots=("input_missing_net_slots", "sum"),
        input_missing_solar_slots=("input_missing_solar_slots", "sum"),
        unresolved_net_slots_replaced_with_zero=(
            "unresolved_net_slots_replaced_with_zero", "sum"
        ),
        unresolved_solar_slots_replaced_with_zero=(
            "unresolved_solar_slots_replaced_with_zero", "sum"
        ),
        nonfinite_feature_values_replaced_with_zero=(
            "nonfinite_feature_values_replaced_with_zero", "sum"
        ),
    )
)

feature_quality_rows = []
for record in partition_index.itertuples(index=False):
    features = pd.read_parquet(record.candidate_path, columns=FEATURE_COLUMNS)
    for feature in FEATURE_COLUMNS:
        values = features[feature].to_numpy(dtype=float)
        feature_quality_rows.append(
            {
                "dataset": record.dataset,
                "substation_id": record.substation_id,
                "feature": feature,
                "rows": len(values),
                "finite_rows": int(np.isfinite(values).sum()),
                "minimum": float(np.nanmin(values)),
                "maximum": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
            }
        )
feature_quality_detail = pd.DataFrame(feature_quality_rows)
feature_quality = (
    feature_quality_detail.groupby(["dataset", "feature"], as_index=False)
    .agg(
        rows=("rows", "sum"),
        finite_rows=("finite_rows", "sum"),
        minimum=("minimum", "min"),
        maximum=("maximum", "max"),
        mean_of_substation_means=("mean", "mean"),
    )
)
feature_quality["nonfinite_rows"] = feature_quality["rows"] - feature_quality["finite_rows"]
assert feature_quality["nonfinite_rows"].eq(0).all()

expected_days = CONFIG["datasets"]["expected_substation_days"]
join_audit_rows = []
for dataset, expected in expected_days.items():
    subset = audit_frame.loc[audit_frame["dataset"].eq(dataset)]
    duplicate_keys = int(subset.duplicated(["substation_id", "date"]).sum())
    join_audit_rows.append(
        {
            "dataset": dataset,
            "expected_final_substation_days": expected,
            "cached_substation_days": len(subset),
            "duplicate_cache_keys": duplicate_keys,
            "missing_cache_keys": expected - len(subset),
            "positive_day_labels": int(subset["true_day"].sum()),
            "sure_substation_days": int(subset["confidence"].eq("sure").sum()),
            "unsure_substation_days": int(subset["confidence"].eq("unsure").sum()),
        }
    )
join_audit = pd.DataFrame(join_audit_rows)
assert join_audit["missing_cache_keys"].eq(0).all()
assert join_audit["duplicate_cache_keys"].eq(0).all()

COUNTS_PATH = OUTPUT_DIRS["tables"] / "table01_candidate_counts.csv"
QUALITY_PATH = OUTPUT_DIRS["tables"] / "table02_feature_quality_summary.csv"
JOIN_PATH = OUTPUT_DIRS["tables"] / "table03_dataset_and_label_join_audit.csv"
write_csv(candidate_counts, COUNTS_PATH)
write_csv(feature_quality, QUALITY_PATH)
write_csv(join_audit, JOIN_PATH)

display(candidate_counts)
display(feature_quality)
display(join_audit)
"""
        ),
        markdown(
            """
## 6. Findings, Limitations, And Manifest

The cache is complete only if all 13,571 Alpha/Beta substation-days are present,
all F1-F9 values are finite, and no key is duplicated. Physical feature scaling
uses no labels. Reviewer confidence is retained only in the day audit so later
notebooks can restrict training to Beta sure days and report Beta sure versus
Beta all after held-out prediction.

This candidate universe is a development artifact rather than an external test.
Its purpose is to let each later subset or weighted model select its own best
window consistently, resolving the earlier fixed-window cache limitation.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [COUNTS_PATH, QUALITY_PATH, JOIN_PATH]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[
        PATHS.config,
        PATHS.final_data / "dataset_alpha.parquet",
        PATHS.final_data / "dataset_beta.parquet",
    ],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "candidate_rows": candidate_metadata.num_rows,
        "substation_days": day_metadata.num_rows,
        "substation_partitions": len(partition_index),
    },
)
manifest["local_intermediates"] = [
    {
        "path": str(CANDIDATE_CACHE.relative_to(PATHS.article)),
        "bytes": CANDIDATE_CACHE.stat().st_size,
        "git_policy": "local_only_reproducible_cache",
    },
    {
        "path": str(DAY_INPUT_CACHE.relative_to(PATHS.article)),
        "bytes": DAY_INPUT_CACHE.stat().st_size,
        "git_policy": "local_only_reproducible_cache",
    },
]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)

output_inventory = pd.DataFrame(
    {
        "type": ["candidate cache", "day cache", "table", "table", "table", "manifest"],
        "path": [
            CANDIDATE_CACHE,
            DAY_INPUT_CACHE,
            COUNTS_PATH,
            QUALITY_PATH,
            JOIN_PATH,
            MANIFEST_PATH,
        ],
    }
)
output_inventory["exists"] = output_inventory["path"].map(Path.exists)
output_inventory["bytes"] = output_inventory["path"].map(lambda path: path.stat().st_size)
display(output_inventory)
assert output_inventory["exists"].all()
assert (output_inventory["bytes"] > 0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02b_m9_pbm_candidate_features.ipynb", cells)


def build_02c(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02c m9_pbm Training-Regime Comparison

**Research question.** Does adding simulated Alpha data improve the
substation-held-out performance of the compact physical model on Beta?

This notebook compares three threshold-training regimes while holding the
candidate score fixed as the equal-weight mean of F1 bridge improvement, F3
slope-continuity improvement, and F4 duration plausibility. Each model selects
its own highest-scoring candidate window before the day threshold is applied.

**Inputs:** the 02b candidate and day caches.  
**Outputs:** fold thresholds, day metrics, two compact tables, two figures, and
a reproducibility manifest.  
**Expected runtime:** under five minutes after 02b exists.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Compact-Model Definition

This experiment uses equal weights only. It does not optimise feature weights
and does not train a machine-learning classifier. Those questions are isolated
in later notebooks.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import (  # noqa: E402
    COMPACT_FEATURE_COLUMNS,
    select_best_candidates,
)
from _m9_pbm_plotting import (  # noqa: E402
    plot_regime_metrics,
    plot_regime_thresholds,
)
from _m9_pbm_validation import (  # noqa: E402
    assert_heldout_absent,
    equal_weights,
    metric_rows,
    select_threshold,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "02c_m9_pbm_training_regimes"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
WEIGHTS = equal_weights(COMPACT_FEATURE_COLUMNS)

display(pd.Series(WEIGHTS, name="equal_candidate_weights"))
display(pd.Series(CONFIG["m9_pbm"]["threshold_selection"], name="threshold_selection"))
"""
        ),
        markdown(
            r"""
## 2. Score, Window Selection, And Day Threshold

For candidate window $W$,

$$
Score_{equal}(W)=\frac{F_1(W)+F_3(W)+F_4(W)}{3}.
$$

The selected window and day prediction are

$$
W_d^*=\operatorname*{arg\,max}_{W}Score_{equal}(W),
\qquad
\widehat{RPF}(d)=\mathbb{1}\{Score_{equal}(W_d^*)\geq\tau\}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $W$ | A valid 30-minute to 8-hour candidate window. |
| $d$ | One substation-day. |
| $F_1(W)$ | Bridge-improvement score for $W$. |
| $F_3(W)$ | Slope-continuity-improvement score for $W$. |
| $F_4(W)$ | Duration-plausibility score for $W$. |
| $Score_{equal}(W)$ | Equal-weight compact physical score. |
| $W_d^*$ | Highest-scoring candidate on day $d$. |
| $\tau$ | Threshold selected from training substations only. |

Candidate selection is not classification: every day has a best candidate, but
only a best-candidate score at or above $\tau$ produces a positive RPF day.
"""
        ),
        markdown(
            """
## 3. Reduce The Candidate Cache Once

The 9.77-million-row cache is processed one substation partition at a time. The
result has one selected candidate per Alpha/Beta substation-day. Labels and Beta
confidence are joined only after physical scoring, with a one-to-one assertion.
"""
        ),
        code(
            """
CACHE_ROOT = PATHS.intermediate / "02b_m9_pbm_candidate_features"
PARTITION_DIR = CACHE_ROOT / "_partitions"
DAY_INPUT_CACHE = CACHE_ROOT / "day_input_cache.parquet"
DAILY_CACHE = OUTPUT_DIRS["intermediate"] / "compact_equal_daily_candidates.parquet"

required_inputs = [PARTITION_DIR, DAY_INPUT_CACHE]
assert all(path.exists() for path in required_inputs), "Run Notebook 02b first."

if DAILY_CACHE.exists() and CONFIG["execution"]["resume_validated_intermediates"]:
    daily = pd.read_parquet(DAILY_CACHE)
else:
    selected_parts = []
    candidate_paths = sorted(PARTITION_DIR.glob("*_candidates.parquet"))
    assert len(candidate_paths) == 18
    for candidate_path in candidate_paths:
        candidates = pd.read_parquet(candidate_path)
        selected_parts.append(select_best_candidates(candidates, WEIGHTS))
    selected = pd.concat(selected_parts, ignore_index=True)
    labels = pd.read_parquet(DAY_INPUT_CACHE)
    daily = selected.merge(
        labels[
            [
                "dataset",
                "substation_id",
                "date",
                "true_day",
                "true_interval_count",
                "confidence",
            ]
        ],
        on=["dataset", "substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    assert daily["true_day"].notna().all()
    write_parquet(daily, DAILY_CACHE)

assert len(daily) == sum(CONFIG["datasets"]["expected_substation_days"].values())
assert daily.duplicated(["dataset", "substation_id", "date"]).sum() == 0
display(
    daily.groupby("dataset", as_index=False).agg(
        substation_days=("date", "size"),
        positive_days=("true_day", "sum"),
        substations=("substation_id", "nunique"),
        mean_selected_score=("score", "mean"),
    )
)
"""
        ),
        markdown(
            """
## 4. Three Training Regimes And Leakage Controls

**Beta only.** Hold out one Beta substation. Select the threshold using sure
days from the other seven Beta substations, then predict every day from the
held-out substation.

**Beta plus Alpha.** Use the same held-out Beta fold, but add all Alpha days to
the seven-substation Beta training set. Alpha and Beta receive equal total
influence in the macro threshold objective, so Alpha's larger sample cannot
dominate.

**Alpha only.** Select one threshold using Alpha alone, then transfer it to all
eight Beta substations. No Beta label contributes to this threshold.

For every regime, Beta sure and Beta all are reporting scopes rather than
alternative predictions. Held-out confidence is consulted only after the
predictions exist.
"""
        ),
        code(
            """
alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
beta = daily.loc[daily["dataset"].eq("beta")].copy()
beta_substations = sorted(beta["substation_id"].unique())

threshold_rows = []
prediction_parts = []
for regime in ["beta_only", "beta_plus_alpha"]:
    for heldout_substation in beta_substations:
        beta_training = beta.loc[
            beta["confidence"].eq("sure")
            & ~beta["substation_id"].eq(heldout_substation)
        ].copy()
        assert_heldout_absent(beta_training, heldout_substation)
        if regime == "beta_only":
            training = beta_training
            dataset_balanced = False
        else:
            training = pd.concat([alpha, beta_training], ignore_index=True)
            dataset_balanced = True

        selection = select_threshold(
            training,
            score_column="score",
            dataset_balanced=dataset_balanced,
        )
        evaluation = beta.loc[beta["substation_id"].eq(heldout_substation)].copy()
        evaluation["regime"] = regime
        evaluation["heldout_substation"] = heldout_substation
        evaluation["threshold"] = selection.threshold
        evaluation["predicted_day"] = evaluation["score"].ge(selection.threshold)
        prediction_parts.append(evaluation)
        threshold_rows.append(
            {
                "regime": regime,
                "heldout_substation": heldout_substation,
                "training_alpha_substations": 0 if regime == "beta_only" else 10,
                "training_beta_substations": 7,
                "training_beta_confidence": "sure_only",
                "dataset_balanced": dataset_balanced,
                "training_rows": len(training),
                "threshold": selection.threshold,
                **selection.metrics,
            }
        )

# Pure Alpha-to-Beta transfer uses no Beta label in threshold selection.
alpha_selection = select_threshold(alpha, score_column="score", dataset_balanced=False)
alpha_transfer = beta.copy()
alpha_transfer["regime"] = "alpha_only"
alpha_transfer["heldout_substation"] = alpha_transfer["substation_id"]
alpha_transfer["threshold"] = alpha_selection.threshold
alpha_transfer["predicted_day"] = alpha_transfer["score"].ge(alpha_selection.threshold)
prediction_parts.append(alpha_transfer)
threshold_rows.append(
    {
        "regime": "alpha_only",
        "heldout_substation": "all_beta",
        "training_alpha_substations": 10,
        "training_beta_substations": 0,
        "training_beta_confidence": "not_used",
        "dataset_balanced": False,
        "training_rows": len(alpha),
        "threshold": alpha_selection.threshold,
        **alpha_selection.metrics,
    }
)

predictions = pd.concat(prediction_parts, ignore_index=True)
thresholds = pd.DataFrame(threshold_rows)
assert predictions.groupby("regime").size().eq(len(beta)).all()
assert predictions.groupby(["regime", "substation_id", "date"]).size().eq(1).all()
display(thresholds[["regime", "heldout_substation", "training_rows", "threshold", "macro_f1"]])
"""
        ),
        markdown(
            """
## 5. Held-Out Beta Metrics

The paper headline is pooled precision, recall, and F1 on held-out **Beta sure**
days. Beta all is retained as a secondary sensitivity analysis. Macro-substation
rows average the eight independently computed substation metrics and are shown
beside pooled results so one large substation cannot hide poor transfer.
"""
        ),
        code(
            """
metric_parts = []
for regime, regime_frame in predictions.groupby("regime", sort=False):
    for confidence_scope, evaluation in [
        ("beta_sure", regime_frame.loc[regime_frame["confidence"].eq("sure")]),
        ("beta_all", regime_frame),
    ]:
        rows = metric_rows(evaluation)
        rows.insert(0, "confidence_scope", confidence_scope)
        rows.insert(0, "regime", regime)
        metric_parts.append(rows)
day_metrics = pd.concat(metric_parts, ignore_index=True)

headline = day_metrics.loc[
    day_metrics["confidence_scope"].eq("beta_sure")
    & day_metrics["aggregation"].isin(["pooled", "macro_substation"])
].copy()
by_substation = day_metrics.loc[day_metrics["aggregation"].eq("substation")].copy()

METRICS_PATH = OUTPUT_DIRS["metrics"] / "01_day_metrics.csv"
THRESHOLDS_PATH = OUTPUT_DIRS["metrics"] / "02_thresholds_by_fold.csv"
HEADLINE_PATH = OUTPUT_DIRS["tables"] / "table01_regime_headline_metrics.csv"
SUBSTATION_PATH = OUTPUT_DIRS["tables"] / "table02_regime_metrics_by_substation.csv"
PREDICTIONS_PATH = OUTPUT_DIRS["intermediate"] / "daily_regime_predictions.parquet"

write_csv(day_metrics, METRICS_PATH)
write_csv(thresholds, THRESHOLDS_PATH)
write_csv(headline, HEADLINE_PATH)
write_csv(by_substation, SUBSTATION_PATH)
write_parquet(predictions, PREDICTIONS_PATH)

display(headline[["regime", "aggregation", "support", "precision", "recall", "f1"]])
"""
        ),
        markdown(
            """
## 6. Figures

The first figure compares held-out pooled Beta-sure scores. The second exposes
fold-to-fold threshold variation for the two Beta LOSO regimes; Alpha-only has
one transfer threshold and is therefore reported in the threshold table rather
than as a misleading eight-point line.
"""
        ),
        code(
            """
FIGURE_METRICS = OUTPUT_DIRS["figures"] / "fig01_regime_precision_recall_f1.png"
FIGURE_THRESHOLDS = OUTPUT_DIRS["figures"] / "fig02_thresholds_by_heldout_substation.png"
pooled_sure = headline.loc[headline["aggregation"].eq("pooled")]
plot_regime_metrics(pooled_sure, FIGURE_METRICS)
plot_regime_thresholds(thresholds, FIGURE_THRESHOLDS)
display(FIGURE_METRICS)
display(FIGURE_THRESHOLDS)
"""
        ),
        markdown(
            """
## 7. Interpretation And Limitations

This comparison isolates the value of training data under one fixed compact
model. It does not claim that Beta is an untouched external dataset: feature
choice was informed by previous Beta development. LOSO nevertheless prevents
the held-out substation's labels from selecting its threshold.

The candidate cache is newly self-consistent, so these results may differ from
earlier fixed-window numbers. Later notebooks use the same candidate-selection
rule rather than forcing agreement with historical values.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [
    METRICS_PATH,
    THRESHOLDS_PATH,
    HEADLINE_PATH,
    SUBSTATION_PATH,
    FIGURE_METRICS,
    FIGURE_THRESHOLDS,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[
        PATHS.config,
        CACHE_ROOT / "candidate_feature_cache.parquet",
        DAY_INPUT_CACHE,
    ],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "daily_selected_candidates": len(daily),
        "heldout_predictions_per_regime": len(beta),
        "regimes": predictions["regime"].nunique(),
        "threshold_rows": len(thresholds),
    },
)
manifest["local_intermediates"] = [
    str(DAILY_CACHE.relative_to(PATHS.article)),
    str(PREDICTIONS_PATH.relative_to(PATHS.article)),
]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)

inventory = pd.DataFrame(
    {"path": [*MANIFEST_OUTPUTS, PREDICTIONS_PATH, MANIFEST_PATH]}
)
inventory["exists"] = inventory["path"].map(Path.exists)
inventory["bytes"] = inventory["path"].map(lambda path: path.stat().st_size)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02c_m9_pbm_training_regimes.ipynb", cells)


def build_02d(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02d m9_pbm Complete Physical-Feature Ablation

**Research question.** How much predictive performance is retained when the
nine-feature physical score is reduced to a smaller, interpretable subset?

This notebook evaluates all $2^9-1=511$ nonempty feature subsets. Every subset
uses equal, unit-sum weights, selects its own best candidate window on every
day, and is evaluated with Beta-plus-Alpha leave-one-substation-out threshold
selection. This is an exploratory model-development comparison, not an untouched
external test.

**Inputs:** the 02b candidate/day caches.  
**Outputs:** the complete 511-row ranking, 4,088 fold thresholds, four compact
tables, three figures, and a manifest.  
**Expected runtime:** approximately 5-20 minutes after 02b exists.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Ablation Contract

Feature names in the tables retain both their number and physical meaning.
Short forms such as F1+F3+F4 are used only in dense figure labels. The candidate
score is divided by feature count, so every subset remains on a comparable
roughly unit scale; this scaling does not change candidate ranking or thresholded
predictions relative to an equal-weight sum.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import FEATURE_COLUMNS, maximum_subset_scores  # noqa: E402
from _m9_pbm_plotting import (  # noqa: E402
    plot_ablation_by_feature_count,
    plot_ablation_feature_evidence,
    plot_top_ablation_subsets,
)
from _m9_pbm_validation import (  # noqa: E402
    all_nonempty_feature_subsets,
    assert_heldout_absent,
    binary_metrics,
    metric_rows,
    select_threshold,
    subset_equal_weight_matrix,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "02d_m9_pbm_feature_ablation"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)

DEFINITIONS = all_nonempty_feature_subsets(FEATURE_COLUMNS)
WEIGHT_MATRIX = subset_equal_weight_matrix(DEFINITIONS, FEATURE_COLUMNS)
SCORE_COLUMNS = [f"subset_{mask:03d}" for mask in DEFINITIONS["subset_mask"]]
DEFINITIONS["score_column"] = SCORE_COLUMNS
DEFINITIONS["feature_set_short"] = [
    "+".join(
        f"F{index + 1}"
        for index, feature in enumerate(FEATURE_COLUMNS)
        if row[f"includes_{feature}"]
    )
    for _, row in DEFINITIONS.iterrows()
]

assert len(DEFINITIONS) == 511
assert DEFINITIONS["subset_mask"].nunique() == 511
assert np.allclose(WEIGHT_MATRIX.sum(axis=1), 1.0)
display(DEFINITIONS[["subset_mask", "feature_count", "feature_set", "feature_set_short"]].head())
"""
        ),
        markdown(
            r"""
## 2. Self-Consistent Subset Scoring

For subset $A$ with $|A|$ active features,

$$
Score_A(W)=\frac{1}{|A|}\sum_{i\in A}F_i(W),
\qquad
W_{d,A}^*=\operatorname*{arg\,max}_{W}Score_A(W).
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $A$ | One nonempty subset of the nine physical features. |
| $|A|$ | Number of active features in that subset. |
| $F_i(W)$ | Physical feature $i$ for candidate window $W$. |
| $Score_A(W)$ | Equal-weight mean physical score for candidate $W$. |
| $W_{d,A}^*$ | Candidate selected by subset $A$ on substation-day $d$. |

The matrix calculation below processes 32 days at a time. It never forms a
9.77-million by 511 matrix in memory. Only the maximum candidate score for each
day and subset is retained for threshold evaluation.
"""
        ),
        code(
            """
CACHE_ROOT = PATHS.intermediate / "02b_m9_pbm_candidate_features"
PARTITION_DIR = CACHE_ROOT / "_partitions"
DAY_INPUT_CACHE = CACHE_ROOT / "day_input_cache.parquet"
SUBSET_DAILY_CACHE = OUTPUT_DIRS["intermediate"] / "all_subset_daily_scores.parquet"
assert PARTITION_DIR.exists() and DAY_INPUT_CACHE.exists(), "Run Notebook 02b first."

if SUBSET_DAILY_CACHE.exists() and CONFIG["execution"]["resume_validated_intermediates"]:
    daily_scores = pd.read_parquet(SUBSET_DAILY_CACHE)
else:
    score_parts = []
    candidate_paths = sorted(PARTITION_DIR.glob("*_candidates.parquet"))
    assert len(candidate_paths) == 18
    for candidate_path in candidate_paths:
        candidates = pd.read_parquet(
            candidate_path,
            columns=[
                "dataset",
                "substation_id",
                "date",
                "candidate_id",
                *FEATURE_COLUMNS,
            ],
        )
        keys, maxima = maximum_subset_scores(
            candidates,
            feature_columns=FEATURE_COLUMNS,
            weight_matrix=WEIGHT_MATRIX,
            batch_days=32,
        )
        score_parts.append(
            pd.concat(
                [keys, pd.DataFrame(maxima, columns=SCORE_COLUMNS)],
                axis=1,
            )
        )
        print(f"Scored {candidate_path.stem}", flush=True)
    daily_scores = pd.concat(score_parts, ignore_index=True)
    labels = pd.read_parquet(DAY_INPUT_CACHE)
    daily_scores = daily_scores.merge(
        labels[["dataset", "substation_id", "date", "true_day", "confidence"]],
        on=["dataset", "substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    write_parquet(daily_scores, SUBSET_DAILY_CACHE)

assert len(daily_scores) == sum(CONFIG["datasets"]["expected_substation_days"].values())
assert daily_scores[SCORE_COLUMNS].notna().all().all()
display(
    pd.Series(
        {
            "substation_days": len(daily_scores),
            "feature_subsets": len(SCORE_COLUMNS),
            "daily_score_values": len(daily_scores) * len(SCORE_COLUMNS),
        },
        name="ablation_cache",
    )
)
"""
        ),
        markdown(
            """
## 3. Beta-Plus-Alpha LOSO Threshold Selection

For each subset and outer held-out Beta substation, the threshold is selected
from all Alpha substations plus sure days from the other seven Beta substations.
Alpha and Beta receive equal total weight in the macro-substation objective.
The held-out Beta substation is predicted once, and its confidence is used only
afterward to create Beta sure and Beta all reports.
"""
        ),
        code(
            """
base_columns = ["dataset", "substation_id", "date", "true_day", "confidence"]
alpha_mask = daily_scores["dataset"].eq("alpha")
beta_frame = daily_scores.loc[daily_scores["dataset"].eq("beta"), base_columns].copy()
beta_frame = beta_frame.reset_index(drop=True)
beta_substations = sorted(beta_frame["substation_id"].unique())
sure_mask = beta_frame["confidence"].eq("sure").to_numpy()

metric_rows_all = []
threshold_rows = []
for definition in DEFINITIONS.itertuples(index=False):
    score_column = definition.score_column
    alpha = daily_scores.loc[alpha_mask, base_columns].copy()
    alpha["score"] = daily_scores.loc[alpha_mask, score_column].to_numpy()
    beta_scores = daily_scores.loc[~alpha_mask, score_column].to_numpy(dtype=float)
    predictions = np.zeros(len(beta_frame), dtype=bool)

    for heldout_substation in beta_substations:
        heldout_mask = beta_frame["substation_id"].eq(heldout_substation).to_numpy()
        beta_training = beta_frame.loc[sure_mask & ~heldout_mask].copy()
        beta_training["score"] = beta_scores[sure_mask & ~heldout_mask]
        assert_heldout_absent(beta_training, heldout_substation)
        training = pd.concat([alpha, beta_training], ignore_index=True)
        selection = select_threshold(
            training,
            score_column="score",
            dataset_balanced=True,
        )
        predictions[heldout_mask] = beta_scores[heldout_mask] >= selection.threshold

        fold_sure = beta_frame.loc[heldout_mask & sure_mask].copy()
        fold_sure["predicted_day"] = predictions[heldout_mask & sure_mask]
        fold_metrics = binary_metrics(fold_sure["true_day"], fold_sure["predicted_day"])
        threshold_rows.append(
            {
                "subset_mask": definition.subset_mask,
                "feature_count": definition.feature_count,
                "feature_set": definition.feature_set,
                "feature_set_short": definition.feature_set_short,
                "heldout_substation": heldout_substation,
                "threshold": selection.threshold,
                "training_rows": len(training),
                "training_macro_f1": selection.metrics["macro_f1"],
                **{f"heldout_sure_{key}": value for key, value in fold_metrics.items()},
            }
        )

    evaluation = beta_frame.copy()
    evaluation["predicted_day"] = predictions
    sure_evaluation = evaluation.loc[evaluation["confidence"].eq("sure")]
    sure_rows = metric_rows(sure_evaluation)
    all_rows = metric_rows(evaluation)
    sure_pooled = sure_rows.loc[sure_rows["aggregation"].eq("pooled")].iloc[0]
    sure_macro = sure_rows.loc[sure_rows["aggregation"].eq("macro_substation")].iloc[0]
    all_pooled = all_rows.loc[all_rows["aggregation"].eq("pooled")].iloc[0]
    metric_rows_all.append(
        {
            "subset_mask": definition.subset_mask,
            "feature_count": definition.feature_count,
            "feature_set": definition.feature_set,
            "feature_set_short": definition.feature_set_short,
            **{f"beta_sure_{key}": sure_pooled[key] for key in [
                "support", "positive_support", "tp", "fp", "fn", "tn",
                "precision", "recall", "f1"
            ]},
            **{f"beta_sure_macro_{key}": sure_macro[key] for key in [
                "precision", "recall", "f1"
            ]},
            **{f"beta_all_{key}": all_pooled[key] for key in [
                "support", "positive_support", "tp", "fp", "fn", "tn",
                "precision", "recall", "f1"
            ]},
        }
    )
    if definition.subset_mask % 50 == 0:
        print(f"Evaluated {definition.subset_mask} of 511 subset masks", flush=True)

ablation_metrics = pd.DataFrame(metric_rows_all).sort_values(
    ["beta_sure_f1", "beta_sure_precision", "beta_sure_recall"],
    ascending=False,
    kind="mergesort",
).reset_index(drop=True)
thresholds = pd.DataFrame(threshold_rows)
assert len(ablation_metrics) == 511
assert len(thresholds) == 511 * 8
display(ablation_metrics.head(20))
"""
        ),
        markdown(
            """
## 4. Best Subsets And Feature Evidence

The best-by-size table answers how many features are needed. Top-subset
frequency asks which features recur among strong models. Paired marginal effects
compare every nonempty subset that excludes a feature with the otherwise
identical subset that includes it. These summaries distinguish a feature that
is broadly helpful from one that is especially useful in a particular
interaction.
"""
        ),
        code(
            """
best_by_count = (
    ablation_metrics.sort_values(
        ["feature_count", "beta_sure_f1", "beta_sure_precision"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    .drop_duplicates("feature_count", keep="first")
    .sort_values("feature_count")
)

metric_by_mask = ablation_metrics.set_index("subset_mask")["beta_sure_f1"]
feature_rows = []
top_10 = set(ablation_metrics.head(10)["subset_mask"])
top_25 = set(ablation_metrics.head(25)["subset_mask"])
top_50 = set(ablation_metrics.head(50)["subset_mask"])
for feature_number, feature in enumerate(FEATURE_COLUMNS, start=1):
    bit = 1 << (feature_number - 1)
    deltas = []
    for subset_mask in range(1, 512):
        if subset_mask & bit:
            continue
        deltas.append(metric_by_mask.loc[subset_mask | bit] - metric_by_mask.loc[subset_mask])
    feature_rows.append(
        {
            "feature_number": feature_number,
            "feature": feature,
            "feature_short": f"F{feature_number}",
            "top_10_frequency_pct": 100 * sum(mask & bit > 0 for mask in top_10) / 10,
            "top_25_frequency_pct": 100 * sum(mask & bit > 0 for mask in top_25) / 25,
            "top_50_frequency_pct": 100 * sum(mask & bit > 0 for mask in top_50) / 50,
            "paired_comparisons": len(deltas),
            "mean_paired_delta_f1": float(np.mean(deltas)),
            "median_paired_delta_f1": float(np.median(deltas)),
            "positive_paired_delta_pct": 100 * float(np.mean(np.asarray(deltas) > 0)),
        }
    )
feature_evidence = pd.DataFrame(feature_rows)
paired_effects = feature_evidence[
    [
        "feature_number", "feature", "paired_comparisons", "mean_paired_delta_f1",
        "median_paired_delta_f1", "positive_paired_delta_pct"
    ]
].copy()

compact_mask = (1 << 0) | (1 << 2) | (1 << 3)
compact_current = ablation_metrics.loc[ablation_metrics["subset_mask"].eq(compact_mask)].iloc[0]
legacy = CONFIG["m9_pbm"]["ablation"]["legacy_fixed_window_compact_anchor"]
best_current = ablation_metrics.iloc[0]
compact_comparison = pd.DataFrame(
    [
        {
            "comparison": "self_consistent_F1_F3_F4",
            "feature_set": compact_current["feature_set"],
            "precision": compact_current["beta_sure_precision"],
            "recall": compact_current["beta_sure_recall"],
            "f1": compact_current["beta_sure_f1"],
            "status": "primary_recomputed",
        },
        {
            "comparison": "best_self_consistent_subset",
            "feature_set": best_current["feature_set"],
            "precision": best_current["beta_sure_precision"],
            "recall": best_current["beta_sure_recall"],
            "f1": best_current["beta_sure_f1"],
            "status": "primary_recomputed",
        },
        {
            "comparison": "legacy_fixed_window_F1_F3_F4_anchor",
            "feature_set": compact_current["feature_set"],
            "precision": legacy["precision"],
            "recall": legacy["recall"],
            "f1": legacy["f1"],
            "status": legacy["source"],
        },
    ]
)
display(
    best_by_count[
        [
            "feature_count",
            "feature_set_short",
            "beta_sure_precision",
            "beta_sure_recall",
            "beta_sure_f1",
        ]
    ]
)
display(feature_evidence)
display(compact_comparison)
"""
        ),
        markdown(
            """
## 5. Write Complete Results And Figures

The full 511-row ranking is compact enough for Git. The wide day-by-subset score
cache is reproducible and remains local. The legacy compact row is explicitly
marked as a recorded fixed-window regression anchor; it is not mixed into the
new ranking.
"""
        ),
        code(
            """
METRICS_PATH = OUTPUT_DIRS["metrics"] / "01_all_511_subset_metrics.csv"
THRESHOLDS_PATH = OUTPUT_DIRS["metrics"] / "02_thresholds_by_subset_and_fold.csv"
BEST_PATH = OUTPUT_DIRS["tables"] / "table01_best_by_feature_count.csv"
FREQUENCY_PATH = OUTPUT_DIRS["tables"] / "table02_feature_frequency.csv"
EFFECT_PATH = OUTPUT_DIRS["tables"] / "table03_paired_marginal_effects.csv"
COMPACT_PATH = OUTPUT_DIRS["tables"] / "table04_compact_model_comparison.csv"
write_csv(ablation_metrics, METRICS_PATH)
write_csv(thresholds, THRESHOLDS_PATH)
write_csv(best_by_count, BEST_PATH)
write_csv(feature_evidence, FREQUENCY_PATH)
write_csv(paired_effects, EFFECT_PATH)
write_csv(compact_comparison, COMPACT_PATH)

FIGURE_COUNT = OUTPUT_DIRS["figures"] / "fig01_f1_by_feature_count.png"
FIGURE_TOP = OUTPUT_DIRS["figures"] / "fig02_top_subset_performance.png"
FIGURE_FEATURE = OUTPUT_DIRS["figures"] / "fig03_feature_frequency_and_marginal_effect.png"
plot_ablation_by_feature_count(ablation_metrics, best_by_count, FIGURE_COUNT)
plot_top_ablation_subsets(ablation_metrics, FIGURE_TOP)
plot_ablation_feature_evidence(feature_evidence, FIGURE_FEATURE)
display(FIGURE_COUNT)
display(FIGURE_TOP)
display(FIGURE_FEATURE)
"""
        ),
        markdown(
            """
## 6. Interpretation And Limitations

This ablation is the complete equal-weight subset search for the newly
self-consistent candidate cache. The central interpretation should compare the
compact F1/F3/F4 result with the best larger subsets and retain the distinction
between broad marginal usefulness and interaction-specific value.

Because the feature family and compact subset were informed by earlier Beta
development, even outer-held-substation predictions are development evidence.
An independent dataset remains necessary for final external validation.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [
    METRICS_PATH, THRESHOLDS_PATH, BEST_PATH, FREQUENCY_PATH, EFFECT_PATH,
    COMPACT_PATH, FIGURE_COUNT, FIGURE_TOP, FIGURE_FEATURE,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[
        PATHS.config,
        CACHE_ROOT / "candidate_feature_cache.parquet",
        DAY_INPUT_CACHE,
    ],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "feature_subsets": len(ablation_metrics),
        "outer_folds": len(thresholds),
        "daily_subset_scores": len(daily_scores) * len(SCORE_COLUMNS),
    },
)
manifest["local_intermediates"] = [str(SUBSET_DAILY_CACHE.relative_to(PATHS.article))]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
inventory = pd.DataFrame({"path": [*MANIFEST_OUTPUTS, SUBSET_DAILY_CACHE, MANIFEST_PATH]})
inventory["exists"] = inventory["path"].map(Path.exists)
inventory["bytes"] = inventory["path"].map(lambda path: path.stat().st_size)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02d_m9_pbm_feature_ablation.ipynb", cells)


def build_02e(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02e m9_pbm Nested Weight Optimisation

**Research question.** Which positive unit-sum weights for F1 bridge, F3 slope
continuity, and F4 duration give the most reliable held-out-substation result?

This notebook compares a 171-point grid and 1,000 seeded random simplex samples.
Weight and threshold selection are nested inside each outer Beta
leave-one-substation-out fold. Alpha is excluded completely. The final model
remains a deterministic physical score, not a machine-learning classifier.

**Inputs:** Beta candidate/day caches from 02b.  
**Outputs:** search results, nested outer metrics, selected weights, two figures,
the final model artifact, the Beta-B outer-fold artifact, and a manifest.  
**Expected runtime:** approximately 10-30 minutes after 02b exists.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Search Space

Every weight is at least 0.05 and the three weights sum to one. The grid uses a
0.05 step. Random samples use seed 9 and the transformation
`0.05 + 0.85 * Dirichlet(1,1,1)`. The exact equal-weight vector is included as
a separate baseline because one third is not on the 0.05 grid.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import (  # noqa: E402
    COMPACT_FEATURE_COLUMNS,
    maximum_subset_scores,
    select_best_candidates,
)
from _m9_pbm_plotting import plot_selected_weights, plot_weight_simplex  # noqa: E402
from _m9_pbm_validation import (  # noqa: E402
    assert_heldout_absent,
    cross_validated_weight_results,
    metric_rows,
    random_simplex_weights,
    select_best_weight_result,
    select_threshold,
    simplex_grid,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "02e_m9_pbm_weight_optimisation"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
WEIGHT_CONFIG = CONFIG["m9_pbm"]["weight_optimisation"]

grid = simplex_grid(
    step=WEIGHT_CONFIG["grid_step"],
    minimum=WEIGHT_CONFIG["minimum_weight"],
)
grid["search_origin"] = "grid"
random = random_simplex_weights(
    WEIGHT_CONFIG["random_samples"],
    minimum=WEIGHT_CONFIG["minimum_weight"],
    seed=WEIGHT_CONFIG["random_seed"],
)
random["search_origin"] = "random"
equal = pd.DataFrame(
    [{
        "weight_F1": 1 / 3,
        "weight_F3": 1 / 3,
        "weight_F4": 1 / 3,
        "search_origin": "equal",
    }]
)
WEIGHT_DEFINITIONS = pd.concat([equal, grid, random], ignore_index=True)
WEIGHT_DEFINITIONS.insert(
    0, "weight_id", [f"weight_{index:04d}" for index in range(len(WEIGHT_DEFINITIONS))]
)
WEIGHT_DEFINITIONS["score_column"] = WEIGHT_DEFINITIONS["weight_id"]
WEIGHT_MATRIX = WEIGHT_DEFINITIONS[
    ["weight_F1", "weight_F3", "weight_F4"]
].to_numpy(dtype=float)

assert len(grid) == 171 and len(random) == 1_000 and len(WEIGHT_DEFINITIONS) == 1_172
assert np.allclose(WEIGHT_MATRIX.sum(axis=1), 1.0)
assert WEIGHT_MATRIX.min() >= WEIGHT_CONFIG["minimum_weight"] - 1e-12
display(WEIGHT_DEFINITIONS.groupby("search_origin").size().rename("weight_vectors"))
display(pd.Series(WEIGHT_CONFIG, name="weight_optimisation"))
"""
        ),
        markdown(
            r"""
## 2. Weighted Candidate Score

For candidate window $W$,

$$
Score_w(W)=w_1F_1(W)+w_3F_3(W)+w_4F_4(W),
$$

subject to

$$
w_1+w_3+w_4=1,\qquad w_1,w_3,w_4\geq0.05.
$$

The selected candidate is

$$
W_{d,w}^*=\operatorname*{arg\,max}_{W}Score_w(W).
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $W$ | One valid candidate correction window. |
| $d$ | One Beta substation-day. |
| $F_1(W)$ | Bridge-improvement feature. |
| $F_3(W)$ | Slope-continuity-improvement feature. |
| $F_4(W)$ | Duration-plausibility feature. |
| $w_1,w_3,w_4$ | Nonzero, unit-sum physical feature weights. |
| $Score_w(W)$ | Weighted candidate score. |
| $W_{d,w}^*$ | Candidate selected by weight vector $w$ on day $d$. |

Only Beta candidates are scored in this notebook. Alpha cannot influence the
final weights, thresholds, or Beta-B forecast-case artifact.
"""
        ),
        markdown(
            """
## 3. Build The Beta Weight-Score Cache

The 1,172 weight vectors are multiplied against F1/F3/F4 in 32-day batches.
For each Beta day and weight vector, only the maximum candidate score is kept.
The wide cache is local and reproducible; it avoids rescanning candidate rows
during the nested folds.
"""
        ),
        code(
            """
CACHE_ROOT = PATHS.intermediate / "02b_m9_pbm_candidate_features"
PARTITION_DIR = CACHE_ROOT / "_partitions"
DAY_INPUT_CACHE = CACHE_ROOT / "day_input_cache.parquet"
WEIGHT_SCORE_CACHE = OUTPUT_DIRS["intermediate"] / "beta_weight_daily_scores.parquet"
assert PARTITION_DIR.exists() and DAY_INPUT_CACHE.exists(), "Run Notebook 02b first."

if WEIGHT_SCORE_CACHE.exists() and CONFIG["execution"]["resume_validated_intermediates"]:
    daily_scores = pd.read_parquet(WEIGHT_SCORE_CACHE)
else:
    score_parts = []
    beta_candidate_paths = sorted(PARTITION_DIR.glob("beta_*_candidates.parquet"))
    assert len(beta_candidate_paths) == 8
    for candidate_path in beta_candidate_paths:
        candidates = pd.read_parquet(
            candidate_path,
            columns=[
                "dataset", "substation_id", "date", "candidate_id",
                *COMPACT_FEATURE_COLUMNS,
            ],
        )
        keys, maxima = maximum_subset_scores(
            candidates,
            feature_columns=COMPACT_FEATURE_COLUMNS,
            weight_matrix=WEIGHT_MATRIX,
            batch_days=32,
        )
        score_parts.append(
            pd.concat(
                [
                    keys,
                    pd.DataFrame(
                        maxima,
                        columns=WEIGHT_DEFINITIONS["score_column"],
                    ),
                ],
                axis=1,
            )
        )
        print(f"Scored {candidate_path.stem}", flush=True)
    daily_scores = pd.concat(score_parts, ignore_index=True)
    labels = pd.read_parquet(DAY_INPUT_CACHE)
    labels = labels.loc[labels["dataset"].eq("beta")]
    daily_scores = daily_scores.merge(
        labels[["dataset", "substation_id", "date", "true_day", "confidence"]],
        on=["dataset", "substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    write_parquet(daily_scores, WEIGHT_SCORE_CACHE)

assert len(daily_scores) == CONFIG["datasets"]["expected_substation_days"]["beta"]
assert daily_scores[WEIGHT_DEFINITIONS["score_column"]].notna().all().all()
assert daily_scores["dataset"].eq("beta").all()
display(
    pd.Series(
        {
            "beta_substation_days": len(daily_scores),
            "weight_vectors": len(WEIGHT_DEFINITIONS),
            "daily_weight_scores": len(daily_scores) * len(WEIGHT_DEFINITIONS),
        },
        name="weight_score_cache",
    )
)
"""
        ),
        markdown(
            """
## 4. Nested Beta Leave-One-Substation-Out Selection

For each outer held-out Beta substation:

1. remove all of its days and labels;
2. run inner LOSO across the other seven substations;
3. in each inner fold, select the threshold on six sure-only substations and
   evaluate the seventh;
4. aggregate inner macro-substation F1 and use precision, recall, then stable
   weight order as tie-breaks;
5. select the best equal, grid, random, and overall optimised vector;
6. retune the threshold on all seven outer-training substations; and
7. predict the outer substation exactly once.

This nesting keeps the outer substation absent from both weight and threshold
selection. Reviewer confidence filters training days only; outer Beta sure/all
metrics are created after prediction.
"""
        ),
        code(
            """
beta_substations = sorted(daily_scores["substation_id"].unique())
outer_search_parts = []
selected_rows = []
outer_decision_parts = []

for outer_heldout in beta_substations:
    inner_results = cross_validated_weight_results(
        daily_scores,
        WEIGHT_DEFINITIONS,
        excluded_substation=outer_heldout,
    )
    inner_results["selection_context"] = f"outer_holdout_{outer_heldout}"
    outer_search_parts.append(inner_results)

    strategy_candidates = {
        "equal": inner_results.loc[inner_results["search_origin"].eq("equal")],
        "grid": inner_results.loc[inner_results["search_origin"].eq("grid")],
        "random": inner_results.loc[inner_results["search_origin"].eq("random")],
        "optimised": inner_results,
    }
    for strategy, candidates in strategy_candidates.items():
        selected = select_best_weight_result(candidates)
        score_column = WEIGHT_DEFINITIONS.set_index("weight_id").loc[
            selected["weight_id"], "score_column"
        ]
        outer_training_mask = (
            daily_scores["confidence"].eq("sure")
            & ~daily_scores["substation_id"].eq(outer_heldout)
        )
        training = daily_scores.loc[
            outer_training_mask,
            ["dataset", "substation_id", "true_day"],
        ].copy()
        training["score"] = daily_scores.loc[outer_training_mask, score_column].to_numpy()
        assert_heldout_absent(training, outer_heldout)
        threshold_selection = select_threshold(training, score_column="score")

        evaluation = daily_scores.loc[daily_scores["substation_id"].eq(outer_heldout)].copy()
        evaluation["strategy"] = strategy
        evaluation["heldout_substation"] = outer_heldout
        evaluation["weight_id"] = selected["weight_id"]
        evaluation["threshold"] = threshold_selection.threshold
        evaluation["selected_score"] = evaluation[score_column]
        evaluation["predicted_day"] = evaluation["selected_score"].ge(
            threshold_selection.threshold
        )
        outer_decision_parts.append(
            evaluation[
                [
                    "dataset", "substation_id", "date", "true_day", "confidence",
                    "strategy", "heldout_substation", "weight_id", "threshold",
                    "selected_score", "predicted_day",
                ]
            ]
        )
        selected_rows.append(
            {
                "strategy": strategy,
                "heldout_substation": outer_heldout,
                "weight_id": selected["weight_id"],
                "search_origin": selected["search_origin"],
                "weight_F1": selected["weight_F1"],
                "weight_F3": selected["weight_F3"],
                "weight_F4": selected["weight_F4"],
                "inner_macro_precision": selected["inner_macro_precision"],
                "inner_macro_recall": selected["inner_macro_recall"],
                "inner_macro_f1": selected["inner_macro_f1"],
                "outer_training_substations": ",".join(
                    substation for substation in beta_substations
                    if substation != outer_heldout
                ),
                "outer_training_confidence": "sure_only",
                "outer_threshold": threshold_selection.threshold,
            }
        )
    print(f"Completed nested selection for {outer_heldout}", flush=True)

outer_search = pd.concat(outer_search_parts, ignore_index=True)
selected_weights = pd.DataFrame(selected_rows)
outer_decisions = pd.concat(outer_decision_parts, ignore_index=True)
assert len(selected_weights) == 8 * 4
assert outer_decisions.groupby("strategy").size().eq(len(daily_scores)).all()
display(
    selected_weights.loc[selected_weights["strategy"].eq("optimised")][
        [
            "heldout_substation", "search_origin", "weight_F1", "weight_F3",
            "weight_F4", "inner_macro_f1", "outer_threshold"
        ]
    ]
)
"""
        ),
        markdown(
            """
## 5. Outer-Fold Metrics And Final Full-Beta Selection

Outer predictions estimate the complete selection procedure. Separately, the
deployment artifact is fitted after evaluation by running LOSO model selection
across all eight Beta substations, choosing one weight vector, and selecting its
threshold from all sure Beta days. This full-data artifact is not used to score
the outer-fold metrics.
"""
        ),
        code(
            """
outer_metric_parts = []
for strategy, strategy_frame in outer_decisions.groupby("strategy", sort=False):
    for confidence_scope, evaluation in [
        ("beta_sure", strategy_frame.loc[strategy_frame["confidence"].eq("sure")]),
        ("beta_all", strategy_frame),
    ]:
        rows = metric_rows(evaluation)
        rows.insert(0, "confidence_scope", confidence_scope)
        rows.insert(0, "strategy", strategy)
        outer_metric_parts.append(rows)
outer_metrics = pd.concat(outer_metric_parts, ignore_index=True)

full_search = cross_validated_weight_results(daily_scores, WEIGHT_DEFINITIONS)
full_search["selection_context"] = "full_beta_model_selection"
final_selected = select_best_weight_result(full_search)
final_score_column = WEIGHT_DEFINITIONS.set_index("weight_id").loc[
    final_selected["weight_id"], "score_column"
]
final_training = daily_scores.loc[
    daily_scores["confidence"].eq("sure"),
    ["dataset", "substation_id", "true_day"],
].copy()
final_training["score"] = daily_scores.loc[
    daily_scores["confidence"].eq("sure"), final_score_column
].to_numpy()
final_threshold = select_threshold(final_training, score_column="score")

all_search = pd.concat([outer_search, full_search], ignore_index=True)
grid_results = all_search.loc[all_search["search_origin"].eq("grid")].copy()
random_results = all_search.loc[all_search["search_origin"].eq("random")].copy()
equal_results = all_search.loc[all_search["search_origin"].eq("equal")].copy()

comparison = outer_metrics.loc[
    outer_metrics["confidence_scope"].eq("beta_sure")
    & outer_metrics["aggregation"].isin(["pooled", "macro_substation"])
].copy()
stability_rows = []
optimised_weights = selected_weights.loc[selected_weights["strategy"].eq("optimised")]
for feature in ["weight_F1", "weight_F3", "weight_F4"]:
    stability_rows.append(
        {
            "weight": feature,
            "outer_fold_mean": optimised_weights[feature].mean(),
            "outer_fold_std": optimised_weights[feature].std(ddof=1),
            "outer_fold_minimum": optimised_weights[feature].min(),
            "outer_fold_maximum": optimised_weights[feature].max(),
            "full_beta_selected": final_selected[feature],
        }
    )
weight_stability = pd.DataFrame(stability_rows)

display(
    comparison[
        ["strategy", "aggregation", "support", "precision", "recall", "f1"]
    ]
)
display(
    pd.Series(
        {
            "weight_id": final_selected["weight_id"],
            "search_origin": final_selected["search_origin"],
            "weight_F1": final_selected["weight_F1"],
            "weight_F3": final_selected["weight_F3"],
            "weight_F4": final_selected["weight_F4"],
            "full_beta_loso_macro_f1": final_selected["inner_macro_f1"],
            "final_threshold": final_threshold.threshold,
        },
        name="final_full_beta_model",
    )
)
"""
        ),
        markdown(
            """
## 6. Recover Outer Selected Windows For Final Evaluation

The wide score cache stores only each day's maximum score. Notebook 02g also
needs the corresponding window boundaries. Therefore each held-out Beta
partition is rescored once using only the optimised inner-selected weight for
that outer fold. The resulting 2,928-row audit contains exactly one leakage-safe
outer prediction per Beta substation-day.
"""
        ),
        code(
            """
prediction_parts = []
optimised_selection = selected_weights.loc[
    selected_weights["strategy"].eq("optimised")
].set_index("heldout_substation")
day_labels = pd.read_parquet(DAY_INPUT_CACHE)
day_labels = day_labels.loc[day_labels["dataset"].eq("beta")]
for heldout_substation in beta_substations:
    selected = optimised_selection.loc[heldout_substation]
    weights = {
        COMPACT_FEATURE_COLUMNS[0]: float(selected["weight_F1"]),
        COMPACT_FEATURE_COLUMNS[1]: float(selected["weight_F3"]),
        COMPACT_FEATURE_COLUMNS[2]: float(selected["weight_F4"]),
    }
    candidate_path = PARTITION_DIR / f"beta_{heldout_substation}_candidates.parquet"
    candidates = pd.read_parquet(candidate_path)
    best_windows = select_best_candidates(candidates, weights)
    predictions = best_windows.merge(
        day_labels.loc[
            day_labels["substation_id"].eq(heldout_substation),
            [
                "dataset", "substation_id", "date", "true_day",
                "true_interval_count", "confidence",
            ],
        ],
        on=["dataset", "substation_id", "date"],
        how="left",
        validate="one_to_one",
    )
    predictions["heldout_substation"] = heldout_substation
    predictions["weight_id"] = selected["weight_id"]
    predictions["weight_F1"] = selected["weight_F1"]
    predictions["weight_F3"] = selected["weight_F3"]
    predictions["weight_F4"] = selected["weight_F4"]
    predictions["threshold"] = selected["outer_threshold"]
    predictions["predicted_day"] = predictions["score"].ge(selected["outer_threshold"])
    predictions["confidence_margin"] = (
        predictions["score"] - selected["outer_threshold"]
    ).abs()
    prediction_parts.append(predictions)

outer_predictions = pd.concat(prediction_parts, ignore_index=True)
assert len(outer_predictions) == len(daily_scores)
assert outer_predictions.duplicated(["substation_id", "date"]).sum() == 0
assert outer_predictions["heldout_substation"].eq(
    outer_predictions["substation_id"]
).all()
PREDICTIONS_PATH = OUTPUT_DIRS["intermediate"] / "nested_outer_predictions.parquet"
write_parquet(outer_predictions, PREDICTIONS_PATH)
display(outer_predictions.head())
"""
        ),
        markdown(
            """
## 7. Write Tables, Figures, And Model Artifacts

The Beta-B artifact is the exact outer-fold model later applied to Gamma. Its
training-substation list excludes Beta B, and its metadata states that neither
Alpha nor Beta-B labels were used. The separate full-Beta model artifact is the
post-evaluation model intended for general future inference.
"""
        ),
        code(
            """
GRID_PATH = OUTPUT_DIRS["metrics"] / "01_grid_search_results.csv"
RANDOM_PATH = OUTPUT_DIRS["metrics"] / "02_random_search_results.csv"
OUTER_PATH = OUTPUT_DIRS["metrics"] / "03_nested_outer_fold_metrics.csv"
SELECTED_PATH = OUTPUT_DIRS["metrics"] / "04_selected_weights_and_thresholds.csv"
COMPARISON_PATH = OUTPUT_DIRS["tables"] / "table01_equal_vs_grid_vs_random.csv"
STABILITY_PATH = OUTPUT_DIRS["tables"] / "table02_weight_stability.csv"
write_csv(grid_results, GRID_PATH)
write_csv(random_results, RANDOM_PATH)
write_csv(outer_metrics, OUTER_PATH)
write_csv(selected_weights, SELECTED_PATH)
write_csv(comparison, COMPARISON_PATH)
write_csv(weight_stability, STABILITY_PATH)

FIGURE_SIMPLEX = OUTPUT_DIRS["figures"] / "fig01_weight_simplex_performance.png"
FIGURE_FOLDS = OUTPUT_DIRS["figures"] / "fig02_selected_weights_by_fold.png"
plot_weight_simplex(full_search, FIGURE_SIMPLEX)
plot_selected_weights(selected_weights, FIGURE_FOLDS)

final_model_payload = {
    "status": "publication_ready",
    "model": "m9_pbm_compact_optimised_physical_score",
    "model_family": "deterministic_non_ml",
    "features": COMPACT_FEATURE_COLUMNS,
    "weights": {
        "F1_bridge_improvement": float(final_selected["weight_F1"]),
        "F3_slope_continuity_improvement": float(final_selected["weight_F3"]),
        "F4_duration_plausibility": float(final_selected["weight_F4"]),
    },
    "threshold": float(final_threshold.threshold),
    "selection_origin": str(final_selected["search_origin"]),
    "selection_objective": "Beta sure macro-substation F1",
    "training_substations": beta_substations,
    "training_confidence": "sure_only",
    "alpha_used": False,
    "random_seed": WEIGHT_CONFIG["random_seed"],
    "candidate_windows": CONFIG["m9_pbm"]["candidate_windows"],
}
FINAL_MODEL_PATH = write_manifest(
    PATHS, "02e_m9_pbm_final_model.json", final_model_payload
)

beta_b_selected = optimised_selection.loc["beta_B"]
beta_b_training = beta_b_selected["outer_training_substations"].split(",")
assert "beta_B" not in beta_b_training
beta_b_payload = {
    "status": "publication_ready",
    "model": "m9_pbm_beta_B_outer_fold",
    "model_family": "deterministic_non_ml",
    "heldout_substation": "beta_B",
    "features": COMPACT_FEATURE_COLUMNS,
    "weights": {
        "F1_bridge_improvement": float(beta_b_selected["weight_F1"]),
        "F3_slope_continuity_improvement": float(beta_b_selected["weight_F3"]),
        "F4_duration_plausibility": float(beta_b_selected["weight_F4"]),
    },
    "threshold": float(beta_b_selected["outer_threshold"]),
    "training_substations": beta_b_training,
    "training_confidence": "sure_only",
    "heldout_labels_used": False,
    "alpha_used": False,
    "candidate_windows": CONFIG["m9_pbm"]["candidate_windows"],
}
BETA_B_MODEL_PATH = write_manifest(
    PATHS, "02e_m9_pbm_beta_B_outer_fold_model.json", beta_b_payload
)
display(FIGURE_SIMPLEX)
display(FIGURE_FOLDS)
display(pd.Series(beta_b_payload, name="Beta-B outer-fold model"))
"""
        ),
        markdown(
            """
## 8. Interpretation, Leakage Statement, And Manifest

The outer-fold optimised result is the performance estimate for the full nested
selection procedure. The full-Beta artifact is fitted only after that estimate
is complete. Beta labels informed feature-family development, so these results
remain development evidence rather than independent external validation.

No Alpha data enters this notebook. The Gamma case-study model excludes Beta B
from inner weight selection, final threshold selection, and every training-label
operation.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [
    GRID_PATH, RANDOM_PATH, OUTER_PATH, SELECTED_PATH, COMPARISON_PATH,
    STABILITY_PATH, FIGURE_SIMPLEX, FIGURE_FOLDS, FINAL_MODEL_PATH,
    BETA_B_MODEL_PATH,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[
        PATHS.config,
        CACHE_ROOT / "candidate_feature_cache.parquet",
        DAY_INPUT_CACHE,
    ],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "weight_vectors": len(WEIGHT_DEFINITIONS),
        "outer_selected_models": len(selected_weights),
        "outer_predictions": len(outer_predictions),
        "grid_result_rows": len(grid_results),
        "random_result_rows": len(random_results),
    },
)
manifest["local_intermediates"] = [
    str(WEIGHT_SCORE_CACHE.relative_to(PATHS.article)),
    str(PREDICTIONS_PATH.relative_to(PATHS.article)),
]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
inventory = pd.DataFrame(
    {"path": [*MANIFEST_OUTPUTS, WEIGHT_SCORE_CACHE, PREDICTIONS_PATH, MANIFEST_PATH]}
)
inventory["exists"] = inventory["path"].map(Path.exists)
inventory["bytes"] = inventory["path"].map(lambda path: path.stat().st_size)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02e_m9_pbm_weight_optimisation.ipynb", cells)


def build_02f(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02f Three-Feature Machine-Learning Comparison

**Research question.** Can DNN, random forest, or XGBoost improve the day
decision when each receives exactly the same F1/F3/F4 physical evidence?

Every model receives one candidate per day selected by the equal-weight
F1/F3/F4 physical score. The ML model sees only those three feature values and
replaces only the final day decision. Calendar variables, substation identity,
raw load, solar, and F2/F5/F6/F7/F8/F9 are prohibited inputs.

**Inputs:** the 02c equal-selected daily candidate cache and 02e outer physical
predictions.  
**Outputs:** nested ML metrics, selected hyperparameters, two compact tables,
one figure, and a manifest.  
**Expected runtime:** approximately 25-50 minutes; each completed outer fold is
checkpointed locally.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Declared Model Grids

The DNN is a standardised two-hidden-layer `MLPClassifier`. Random forest and
XGBoost use deterministic seeds and weighted training rows. The grids are small
and fixed in configuration: 4 DNN, 6 random-forest, and 4 XGBoost combinations.
No broad exploratory search is launched.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import COMPACT_FEATURE_COLUMNS  # noqa: E402
from _m9_pbm_plotting import plot_physical_vs_ml  # noqa: E402
from _m9_pbm_validation import (  # noqa: E402
    metric_rows,
    ml_hyperparameter_definitions,
    run_nested_ml_outer_experiment,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "02f_m9_pbm_ml_comparison"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
CHECKPOINT_DIR = OUTPUT_DIRS["intermediate"] / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SEED = CONFIG["m9_pbm"]["ml_comparison"]["random_seed"]
DEFINITIONS = ml_hyperparameter_definitions(CONFIG)

assert len(DEFINITIONS) == 14
assert set(DEFINITIONS["model"]) == {"dnn", "random_forest", "xgboost"}
display(
    DEFINITIONS[["model", "hyperparameter_id", "parameters_json"]]
)
"""
        ),
        markdown(
            r"""
## 2. Fair-Comparison Input And Decision

Candidate selection remains the equal physical score

$$
W_d^*=\operatorname*{arg\,max}_{W}
\frac{F_1(W)+F_3(W)+F_4(W)}{3}.
$$

The classifier then estimates

$$
p_d=P(RPF_d=1\mid F_1(W_d^*),F_3(W_d^*),F_4(W_d^*)),
$$

and predicts

$$
\widehat{RPF}(d)=\mathbb{1}\{p_d\geq\tau_{ML}\}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $d$ | One substation-day. |
| $W$ | One valid physical candidate window. |
| $W_d^*$ | Equal-F1/F3/F4 candidate selected before ML. |
| $F_1,F_3,F_4$ | Bridge, slope-continuity, and duration features. |
| $p_d$ | Model-estimated probability of an RPF sign-error day. |
| $\tau_{ML}$ | Probability threshold selected from training data only. |

ML cannot move the window boundaries. This isolates whether a nonlinear day
decision adds value beyond the deterministic physical score.
"""
        ),
        markdown(
            """
## 3. Load And Validate Equal-Selected Daily Features

The selected-candidate cache must contain exactly three allowed model inputs.
Labels and confidence are retained for nested training and reporting, not as
features. The final physical predictions are loaded separately for the fair
comparison table.
"""
        ),
        code(
            """
DAILY_CACHE = (
    PATHS.intermediate
    / "02c_m9_pbm_training_regimes"
    / "compact_equal_daily_candidates.parquet"
)
PHYSICAL_PREDICTIONS = (
    PATHS.intermediate
    / "02e_m9_pbm_weight_optimisation"
    / "nested_outer_predictions.parquet"
)
assert DAILY_CACHE.exists() and PHYSICAL_PREDICTIONS.exists(), "Run 02c and 02e first."

daily = pd.read_parquet(DAILY_CACHE)
alpha = daily.loc[daily["dataset"].eq("alpha")].copy()
beta = daily.loc[daily["dataset"].eq("beta")].copy()
beta_sure = beta.loc[beta["confidence"].eq("sure")].copy()
beta_substations = sorted(beta["substation_id"].unique())
alpha_substations = sorted(alpha["substation_id"].unique())

assert len(COMPACT_FEATURE_COLUMNS) == 3
assert daily[COMPACT_FEATURE_COLUMNS].notna().all().all()
assert len(beta) == CONFIG["datasets"]["expected_substation_days"]["beta"]
display(
    pd.DataFrame(
        [
            {"dataset": "Alpha", "rows": len(alpha), "training_rows": len(alpha)},
            {"dataset": "Beta sure", "rows": len(beta_sure), "training_rows": len(beta_sure)},
            {"dataset": "Beta all", "rows": len(beta), "training_rows": "reporting only"},
        ]
    )
)
"""
        ),
        markdown(
            """
## 4. Nested Beta LOSO For Beta-Only And Beta-Plus-Alpha

For each outer Beta substation, hyperparameters are selected by inner LOSO on
the other seven. Each inner model trains on six Beta substations, plus all Alpha
only in the mixed regime. Its probability threshold is selected on those
training rows before the seventh Beta substation is evaluated. The selected
configuration is then fitted on all seven outer-training Beta substations and
predicts the outer substation once.

Beta training uses sure days only. Outer predictions include all days, and
confidence is consulted only afterward for Beta sure/all reporting. Checkpoints
are written after every completed regime-fold pair.
"""
        ),
        code(
            """
inner_parts = []
selected_parts = []
decision_parts = []
for regime in ["beta_only", "beta_plus_alpha"]:
    for outer_heldout in beta_substations:
        stem = f"{regime}_{outer_heldout}"
        inner_path = CHECKPOINT_DIR / f"{stem}_inner.csv"
        selected_path = CHECKPOINT_DIR / f"{stem}_selected.csv"
        decision_path = CHECKPOINT_DIR / f"{stem}_decisions.parquet"
        reusable = (
            CONFIG["execution"]["resume_validated_intermediates"]
            and inner_path.exists()
            and selected_path.exists()
            and decision_path.exists()
        )
        if reusable:
            inner_results = pd.read_csv(inner_path)
            selected = pd.read_csv(selected_path)
            decisions = pd.read_parquet(decision_path)
        else:
            beta_training = beta_sure.loc[
                ~beta_sure["substation_id"].eq(outer_heldout)
            ].copy()
            training_pool = (
                beta_training
                if regime == "beta_only"
                else pd.concat([alpha, beta_training], ignore_index=True)
            )
            outer_evaluation = beta.loc[
                beta["substation_id"].eq(outer_heldout)
            ].copy()
            inner_substations = [
                substation for substation in beta_substations
                if substation != outer_heldout
            ]
            inner_results, selected, decisions = run_nested_ml_outer_experiment(
                training_pool,
                outer_evaluation,
                DEFINITIONS,
                feature_columns=COMPACT_FEATURE_COLUMNS,
                inner_fold_substations=inner_substations,
                dataset_balanced=regime == "beta_plus_alpha",
                seed=SEED,
                regime=regime,
                outer_identifier=outer_heldout,
            )
            write_csv(inner_results, inner_path)
            write_csv(selected, selected_path)
            write_parquet(decisions, decision_path)
        inner_parts.append(inner_results)
        selected_parts.append(selected)
        decision_parts.append(decisions)
        print(f"{regime} {outer_heldout}: {'reused' if reusable else 'completed'}", flush=True)
"""
        ),
        markdown(
            """
## 5. Alpha-Only Transfer

Alpha-only hyperparameters are selected by leave-one-Alpha-substation-out
validation. The selected model and probability threshold are then fitted using
all Alpha rows and transferred to all Beta substations. No Beta label or
confidence value enters Alpha-only model selection.
"""
        ),
        code(
            """
stem = "alpha_only_all_beta"
inner_path = CHECKPOINT_DIR / f"{stem}_inner.csv"
selected_path = CHECKPOINT_DIR / f"{stem}_selected.csv"
decision_path = CHECKPOINT_DIR / f"{stem}_decisions.parquet"
reusable = (
    CONFIG["execution"]["resume_validated_intermediates"]
    and inner_path.exists()
    and selected_path.exists()
    and decision_path.exists()
)
if reusable:
    alpha_inner = pd.read_csv(inner_path)
    alpha_selected = pd.read_csv(selected_path)
    alpha_decisions = pd.read_parquet(decision_path)
else:
    alpha_inner, alpha_selected, alpha_decisions = run_nested_ml_outer_experiment(
        alpha,
        beta,
        DEFINITIONS,
        feature_columns=COMPACT_FEATURE_COLUMNS,
        inner_fold_substations=alpha_substations,
        dataset_balanced=False,
        seed=SEED,
        regime="alpha_only",
        outer_identifier="all_beta",
    )
    write_csv(alpha_inner, inner_path)
    write_csv(alpha_selected, selected_path)
    write_parquet(alpha_decisions, decision_path)
inner_parts.append(alpha_inner)
selected_parts.append(alpha_selected)
decision_parts.append(alpha_decisions)

inner_results = pd.concat(inner_parts, ignore_index=True)
selected_hyperparameters = pd.concat(selected_parts, ignore_index=True)
ml_decisions = pd.concat(decision_parts, ignore_index=True)
assert len(selected_hyperparameters) == 8 * 2 * 3 + 3
assert ml_decisions.groupby(["regime", "model"]).size().eq(len(beta)).all()
display(
    selected_hyperparameters[
        [
            "regime", "outer_identifier", "model", "hyperparameter_id",
            "inner_macro_f1", "outer_threshold"
        ]
    ]
)
"""
        ),
        markdown(
            """
## 6. Held-Out Beta Sure And Beta All Metrics

The primary comparison uses pooled held-out Beta-sure precision, recall, and F1.
Macro-substation and per-substation rows are retained. The deterministic
optimised physical model is evaluated from its 02e outer predictions and cannot
be displaced as the final method regardless of ML ranking.
"""
        ),
        code(
            """
metric_parts = []
for (regime, model), model_frame in ml_decisions.groupby(
    ["regime", "model"], sort=False
):
    for confidence_scope, evaluation in [
        ("beta_sure", model_frame.loc[model_frame["confidence"].eq("sure")]),
        ("beta_all", model_frame),
    ]:
        rows = metric_rows(evaluation)
        rows.insert(0, "confidence_scope", confidence_scope)
        rows.insert(0, "model", model)
        rows.insert(0, "regime", regime)
        metric_parts.append(rows)
ml_metrics = pd.concat(metric_parts, ignore_index=True)

physical = pd.read_parquet(PHYSICAL_PREDICTIONS)
physical_metric_parts = []
for confidence_scope, evaluation in [
    ("beta_sure", physical.loc[physical["confidence"].eq("sure")]),
    ("beta_all", physical),
]:
    rows = metric_rows(evaluation)
    rows.insert(0, "confidence_scope", confidence_scope)
    rows.insert(0, "model", "m9_pbm_optimised_physical")
    rows.insert(0, "regime", "nested_beta_only")
    physical_metric_parts.append(rows)
physical_metrics = pd.concat(physical_metric_parts, ignore_index=True)

all_metrics = pd.concat([physical_metrics, ml_metrics], ignore_index=True)
comparison = all_metrics.loc[
    all_metrics["confidence_scope"].eq("beta_sure")
    & all_metrics["aggregation"].isin(["pooled", "macro_substation"])
].copy()
by_substation = all_metrics.loc[all_metrics["aggregation"].eq("substation")].copy()

display(
    comparison[
        [
            "regime", "model", "aggregation", "support", "precision", "recall", "f1"
        ]
    ]
)
"""
        ),
        markdown(
            """
## 7. Write Results And Comparison Figure

The figure shows pooled Beta-sure results. Each ML label includes its training
regime, while the physical model appears once using the nested outer predictions
from 02e. Hyperparameter definitions and selected thresholds remain available
in CSV; fitted estimators are intentionally not serialized.
"""
        ),
        code(
            """
METRICS_PATH = OUTPUT_DIRS["metrics"] / "01_ml_nested_metrics.csv"
HYPERPARAMETER_PATH = OUTPUT_DIRS["metrics"] / "02_selected_hyperparameters.csv"
COMPARISON_PATH = OUTPUT_DIRS["tables"] / "table01_physical_vs_ml.csv"
SUBSTATION_PATH = OUTPUT_DIRS["tables"] / "table02_ml_by_substation.csv"
write_csv(ml_metrics, METRICS_PATH)
write_csv(selected_hyperparameters, HYPERPARAMETER_PATH)
write_csv(comparison, COMPARISON_PATH)
write_csv(by_substation, SUBSTATION_PATH)

model_labels = {
    "m9_pbm_optimised_physical": "Physical / nested Beta",
    "dnn": "DNN",
    "random_forest": "RF",
    "xgboost": "XGB",
}
regime_labels = {
    "beta_only": "Beta",
    "beta_plus_alpha": "Beta + Alpha",
    "alpha_only": "Alpha",
    "nested_beta_only": "nested Beta",
}
figure_data = comparison.loc[comparison["aggregation"].eq("pooled")].copy()
figure_data["display_label"] = [
    model_labels[model]
    if model == "m9_pbm_optimised_physical"
    else f"{model_labels[model]} / {regime_labels[regime]}"
    for model, regime in zip(figure_data["model"], figure_data["regime"], strict=True)
]
FIGURE_PATH = OUTPUT_DIRS["figures"] / "fig01_physical_vs_ml_precision_recall_f1.png"
plot_physical_vs_ml(figure_data, FIGURE_PATH)
display(FIGURE_PATH)
"""
        ),
        markdown(
            """
## 8. Interpretation, Leakage Statement, And Manifest

This experiment tests only the final day decision. Candidate generation and
window selection remain the same compact physical process for all models.
Nested folds prevent the outer Beta substation from selecting ML
hyperparameters or thresholds in the Beta-trained regimes. Alpha-only transfer
uses no Beta labels.

The experiment is model-development evidence because Beta informed earlier
feature selection. The deterministic optimised F1/F3/F4 model remains the final
`m9_pbm` method even if an ML row scores higher.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [
    METRICS_PATH, HYPERPARAMETER_PATH, COMPARISON_PATH, SUBSTATION_PATH, FIGURE_PATH,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[PATHS.config, DAILY_CACHE, PHYSICAL_PREDICTIONS],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "hyperparameter_definitions": len(DEFINITIONS),
        "selected_hyperparameter_rows": len(selected_hyperparameters),
        "ml_outer_decisions": len(ml_decisions),
        "ml_metric_rows": len(ml_metrics),
    },
)
manifest["local_intermediates"] = [str(CHECKPOINT_DIR.relative_to(PATHS.article))]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
inventory = pd.DataFrame({"path": [*MANIFEST_OUTPUTS, MANIFEST_PATH]})
inventory["exists"] = inventory["path"].map(Path.exists)
inventory["bytes"] = inventory["path"].map(lambda path: path.stat().st_size)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02f_m9_pbm_ml_comparison.ipynb", cells)


def build_02g(notebook_dir: Path) -> Path:
    cells = [
        markdown(
            """
# 02g Final m9_pbm Evaluation

**Research question.** How well does the final deterministic physical procedure
classify days, localise correction windows, recover correction energy, and
reduce manual review on held-out Beta substations?

This notebook uses only the nested outer predictions from 02e. Each Beta
substation-day was predicted by weights and a threshold selected without that
substation's labels. The notebook expands positive windows to 15-minute flags
and reports Beta sure as primary, with Beta all as a sensitivity analysis.

**Inputs:** final Beta data and 02e nested outer predictions.  
**Outputs:** five metric CSVs, four compact tables, five figures, a local
interval audit, and a manifest.  
**Expected runtime:** under five minutes.
"""
        ),
        markdown(
            """
## 1. Imports, Paths, And Operating-Point Configuration

Confidence coverage levels and the development operating-point constraints are
read from configuration. The recommended point is the largest tested coverage
with auto-accepted precision at least 0.99 and F1 at least 0.95; if no tested
level qualifies, that absence is reported explicitly.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display


def find_notebook_directory() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name == "notebooks" and (candidate / "_m9_pbm_data.py").exists():
            return candidate
        nested = candidate / "publication" / "2_journal_article" / "notebooks"
        if (nested / "_m9_pbm_data.py").exists():
            return nested
    raise FileNotFoundError("Could not locate the journal notebook directory.")


NOTEBOOK_DIR = find_notebook_directory()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _m9_pbm_data import (  # noqa: E402
    load_dataset,
    load_experiment_config,
    manifest_payload,
    output_dirs,
    resolve_paths,
    write_csv,
    write_manifest,
    write_parquet,
)
from _m9_pbm_features import window_iou  # noqa: E402
from _m9_pbm_plotting import (  # noqa: E402
    plot_auto_accept_burden,
    plot_coverage_scores,
    plot_energy_summary,
    plot_final_confusion_matrices,
    plot_window_iou_distribution,
)
from _m9_pbm_validation import (  # noqa: E402
    confidence_coverage_metrics,
    metric_rows,
    recommended_coverage,
)

STARTED_AT = time.time()
ARTICLE_ROOT = NOTEBOOK_DIR.parent
CONFIG = load_experiment_config(ARTICLE_ROOT)
PATHS = resolve_paths(ARTICLE_ROOT, CONFIG)
SLUG = "02g_m9_pbm_final_evaluation"
OUTPUT_DIRS = output_dirs(PATHS, SLUG)
COVERAGE_CONFIG = CONFIG["m9_pbm"]["confidence_coverage"]

display(pd.Series(COVERAGE_CONFIG, name="confidence_coverage"))
"""
        ),
        markdown(
            """
## 2. Build The Held-Out Interval Audit

Every positive day uses the candidate selected by its outer-fold weight vector.
For a negative day, no interval is predicted positive even though the day still
has a highest-scoring candidate. The join must preserve all 280,800 final Beta
rows and exactly one daily prediction key.
"""
        ),
        code(
            """
PREDICTION_PATH = (
    PATHS.intermediate
    / "02e_m9_pbm_weight_optimisation"
    / "nested_outer_predictions.parquet"
)
assert PREDICTION_PATH.exists(), "Run Notebook 02e first."
predictions = pd.read_parquet(PREDICTION_PATH)
beta = load_dataset("beta", article_root=ARTICLE_ROOT, config=CONFIG)
beta["slot"] = beta["timestamp"].dt.hour * 4 + beta["timestamp"].dt.minute // 15

prediction_columns = [
    "substation_id", "date", "left_slot", "right_slot", "score", "threshold",
    "predicted_day", "true_day", "confidence", "confidence_margin",
    "weight_F1", "weight_F3", "weight_F4",
]
day_predictions = predictions[prediction_columns].copy()
day_predictions = day_predictions.rename(
    columns={"true_day": "prediction_true_day", "confidence": "prediction_confidence"}
)
interval_audit = beta.merge(
    day_predictions,
    on=["substation_id", "date"],
    how="left",
    validate="many_to_one",
)
interval_audit["predicted_interval"] = (
    interval_audit["predicted_day"]
    & interval_audit["slot"].between(
        interval_audit["left_slot"], interval_audit["right_slot"]
    )
)

assert len(interval_audit) == len(beta) == 280_800
assert interval_audit["predicted_day"].notna().all()
assert interval_audit["label_day"].eq(interval_audit["prediction_true_day"]).all()
assert interval_audit["confidence"].eq(interval_audit["prediction_confidence"]).all()
assert predictions.duplicated(["substation_id", "date"]).sum() == 0

INTERVAL_AUDIT_PATH = OUTPUT_DIRS["intermediate"] / "heldout_prediction_audit.parquet"
write_parquet(interval_audit, INTERVAL_AUDIT_PATH)
display(
    pd.Series(
        {
            "interval_rows": len(interval_audit),
            "substation_days": len(predictions),
            "predicted_positive_days": int(predictions["predicted_day"].sum()),
            "predicted_positive_intervals": int(interval_audit["predicted_interval"].sum()),
        },
        name="outer_prediction_audit",
    )
)
"""
        ),
        markdown(
            r"""
## 3. Day And Interval Classification Metrics

Day metrics compare $RPF(d)$ with $\widehat{RPF}(d)$. Interval metrics compare
the manual flag $z(t)$ with the expanded predicted flag $\hat z(t)$. Full-day
interval metrics are primary; 06:00-18:00 is a diagnostic scope.

For either level,

$$
Precision=\frac{TP}{TP+FP},\qquad
Recall=\frac{TP}{TP+FN},\qquad
F1=\frac{2\,Precision\,Recall}{Precision+Recall}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $d$ | One Beta substation-day. |
| $t$ | One 15-minute Beta timestamp. |
| $RPF(d)$ | Final manual day label. |
| $\widehat{RPF}(d)$ | Nested outer-fold predicted day label. |
| $z(t)$ | Final manual interval label. |
| $\hat z(t)$ | Predicted interval flag from the selected positive-day window. |
| $TP,FP,FN,TN$ | True-positive, false-positive, false-negative, and true-negative counts. |
"""
        ),
        code(
            """
day_metric_parts = []
for confidence_scope, evaluation in [
    ("beta_sure", predictions.loc[predictions["confidence"].eq("sure")]),
    ("beta_all", predictions),
]:
    rows = metric_rows(evaluation)
    rows.insert(0, "confidence_scope", confidence_scope)
    day_metric_parts.append(rows)
day_metrics = pd.concat(day_metric_parts, ignore_index=True)

interval_metric_parts = []
for confidence_scope, confidence_frame in [
    ("beta_sure", interval_audit.loc[interval_audit["confidence"].eq("sure")]),
    ("beta_all", interval_audit),
]:
    for interval_scope, evaluation in [
        ("full_day", confidence_frame),
        ("daytime_06_18", confidence_frame.loc[confidence_frame["slot"].between(24, 72)]),
    ]:
        rows = metric_rows(
            evaluation,
            truth_column="label_interval",
            prediction_column="predicted_interval",
        )
        rows.insert(0, "interval_scope", interval_scope)
        rows.insert(0, "confidence_scope", confidence_scope)
        interval_metric_parts.append(rows)
interval_metrics = pd.concat(interval_metric_parts, ignore_index=True)

display(
    day_metrics.loc[
        day_metrics["aggregation"].isin(["pooled", "macro_substation"]),
        ["confidence_scope", "aggregation", "support", "precision", "recall", "f1"],
    ]
)
display(
    interval_metrics.loc[
        interval_metrics["aggregation"].eq("pooled"),
        [
            "confidence_scope", "interval_scope", "support", "precision", "recall", "f1"
        ],
    ]
)
"""
        ),
        markdown(
            r"""
## 4. Candidate-Window Intersection Over Union

For true interval set $T_d$ and predicted interval set $P_d$,

$$
IoU(d)=\frac{|T_d\cap P_d|}{|T_d\cup P_d|}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $T_d$ | Set of manually labelled RPF slots on day $d$. |
| $P_d$ | Set of predicted correction slots on day $d$; empty for a negative decision. |
| $|\cdot|$ | Number of 15-minute slots in a set. |
| $IoU(d)$ | Slot overlap divided by slot union. |

`tp_days_only` includes days where truth and prediction are both positive.
`event_days_truth_or_prediction` additionally includes FP and FN days, assigning
IoU zero when exactly one set is empty. Days with both sets empty are outside
both reported scopes.
"""
        ),
        code(
            """
window_rows = []
for (substation, date), group in interval_audit.groupby(
    ["substation_id", "date"], sort=True
):
    true_slots = set(group.loc[group["label_interval"], "slot"].astype(int))
    predicted_slots = set(group.loc[group["predicted_interval"], "slot"].astype(int))
    true_day = bool(group["label_day"].max())
    predicted_day = bool(group["predicted_day"].iloc[0])
    both_windows = bool(true_slots and predicted_slots)
    window_rows.append(
        {
            "substation_id": substation,
            "date": date,
            "confidence": group["confidence"].iloc[0],
            "true_day": true_day,
            "predicted_day": predicted_day,
            "true_interval_count": len(true_slots),
            "predicted_interval_count": len(predicted_slots),
            "window_iou": window_iou(true_slots, predicted_slots),
            "absolute_start_error_minutes": (
                15 * abs(min(predicted_slots) - min(true_slots)) if both_windows else np.nan
            ),
            "absolute_end_error_minutes": (
                15 * abs(max(predicted_slots) - max(true_slots)) if both_windows else np.nan
            ),
        }
    )
window_audit = pd.DataFrame(window_rows)

def summarise_iou(frame, confidence_scope, event_scope, aggregation, substation_id=""):
    values = frame["window_iou"].dropna()
    return {
        "confidence_scope": confidence_scope,
        "event_scope": event_scope,
        "aggregation": aggregation,
        "substation_id": substation_id,
        "support_days": len(values),
        "mean_iou": values.mean() if len(values) else np.nan,
        "median_iou": values.median() if len(values) else np.nan,
        "proportion_iou_at_least_0_50": values.ge(0.50).mean() if len(values) else np.nan,
        "proportion_iou_at_least_0_70": values.ge(0.70).mean() if len(values) else np.nan,
        "median_absolute_start_error_minutes": frame["absolute_start_error_minutes"].median(),
        "median_absolute_end_error_minutes": frame["absolute_end_error_minutes"].median(),
    }

iou_rows = []
for confidence_scope, confidence_frame in [
    ("beta_sure", window_audit.loc[window_audit["confidence"].eq("sure")]),
    ("beta_all", window_audit),
]:
    event_frames = {
        "tp_days_only": confidence_frame.loc[
            confidence_frame["true_day"] & confidence_frame["predicted_day"]
        ],
        "event_days_truth_or_prediction": confidence_frame.loc[
            confidence_frame["true_day"] | confidence_frame["predicted_day"]
        ],
    }
    for event_scope, event_frame in event_frames.items():
        iou_rows.append(
            summarise_iou(event_frame, confidence_scope, event_scope, "pooled")
        )
        for substation, group in event_frame.groupby("substation_id", sort=True):
            iou_rows.append(
                summarise_iou(
                    group, confidence_scope, event_scope, "substation", substation
                )
            )
window_metrics = pd.DataFrame(iou_rows)
display(window_metrics.loc[window_metrics["aggregation"].eq("pooled")])
"""
        ),
        markdown(
            r"""
## 5. Correction-Energy Agreement

At each quarter hour, the energy associated with correcting an incorrectly
positive net-load reading is

$$
e(t)=2\max(y(t),0)\times0.25\ \text{hours}.
$$

Pooled correction-energy IoU is

$$
Energy\ IoU=\frac{\sum_{t:z(t)=1\land\hat z(t)=1}e(t)}
{\sum_{t:z(t)=1\lor\hat z(t)=1}e(t)}.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $y(t)$ | Observed net load in MW. |
| $e(t)$ | Correction magnitude in MWh for one 15-minute interval. |
| $z(t)$ | Manual interval flag. |
| $\hat z(t)$ | Predicted interval flag. |
| $\land,\lor$ | Logical AND and OR. |

Energy precision, recall, and F1 use the same overlap as their numerator and
predicted/manual correction energy as their denominators. Full day is primary;
daytime is retained as an audit.
"""
        ),
        code(
            """
interval_audit["correction_energy_MWh"] = (
    2 * interval_audit["net_load_MW"].clip(lower=0).fillna(0) * 0.25
)
interval_audit["manual_correction_MWh"] = np.where(
    interval_audit["label_interval"], interval_audit["correction_energy_MWh"], 0.0
)
interval_audit["predicted_correction_MWh"] = np.where(
    interval_audit["predicted_interval"], interval_audit["correction_energy_MWh"], 0.0
)
interval_audit["overlap_correction_MWh"] = np.where(
    interval_audit["label_interval"] & interval_audit["predicted_interval"],
    interval_audit["correction_energy_MWh"],
    0.0,
)
interval_audit["union_correction_MWh"] = np.where(
    interval_audit["label_interval"] | interval_audit["predicted_interval"],
    interval_audit["correction_energy_MWh"],
    0.0,
)

def summarise_energy(frame, confidence_scope, interval_scope, aggregation, substation_id=""):
    manual = frame["manual_correction_MWh"].sum()
    predicted = frame["predicted_correction_MWh"].sum()
    overlap = frame["overlap_correction_MWh"].sum()
    union = frame["union_correction_MWh"].sum()
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / manual if manual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence_scope": confidence_scope,
        "interval_scope": interval_scope,
        "aggregation": aggregation,
        "substation_id": substation_id,
        "manual_correction_MWh": manual,
        "predicted_correction_MWh": predicted,
        "overlap_correction_MWh": overlap,
        "union_correction_MWh": union,
        "energy_precision": precision,
        "energy_recall": recall,
        "energy_f1": f1,
        "energy_iou": overlap / union if union else np.nan,
    }

energy_rows = []
for confidence_scope, confidence_frame in [
    ("beta_sure", interval_audit.loc[interval_audit["confidence"].eq("sure")]),
    ("beta_all", interval_audit),
]:
    for interval_scope, evaluation in [
        ("full_day", confidence_frame),
        ("daytime_06_18", confidence_frame.loc[confidence_frame["slot"].between(24, 72)]),
    ]:
        energy_rows.append(
            summarise_energy(evaluation, confidence_scope, interval_scope, "pooled")
        )
        for substation, group in evaluation.groupby("substation_id", sort=True):
            energy_rows.append(
                summarise_energy(
                    group, confidence_scope, interval_scope, "substation", substation
                )
            )
energy_metrics = pd.DataFrame(energy_rows)
display(energy_metrics.loc[energy_metrics["aggregation"].eq("pooled")])
"""
        ),
        markdown(
            r"""
## 6. Confidence Coverage And Manual Review

For each outer prediction,

$$
margin(d)=|Score(W_d^*)-\tau_d|.
$$

**Notation**

| Symbol | Meaning |
|---|---|
| $Score(W_d^*)$ | Selected candidate's unit-sum weighted physical score. |
| $\tau_d$ | Threshold from the fold that held out day $d$'s substation. |
| $margin(d)$ | Distance from the decision boundary; larger means more confident. |

At each coverage level, the largest-margin positive and negative decisions are
auto-accepted. Remaining days are sent to manual review. Metrics therefore
describe only the accepted subset, while the burden table also records how many
days and true RPF days remain for review.
"""
        ),
        code(
            """
sure_predictions = predictions.loc[predictions["confidence"].eq("sure")].copy()
coverage_metrics = confidence_coverage_metrics(
    sure_predictions,
    COVERAGE_CONFIG["levels_pct"],
)
recommended = recommended_coverage(
    coverage_metrics,
    minimum_precision=COVERAGE_CONFIG["minimum_precision"],
    minimum_f1=COVERAGE_CONFIG["minimum_f1"],
)
if recommended is None:
    recommended_table = pd.DataFrame(
        [{
            "qualifying_operating_point_found": False,
            "minimum_precision": COVERAGE_CONFIG["minimum_precision"],
            "minimum_f1": COVERAGE_CONFIG["minimum_f1"],
            "recommended_coverage_pct": np.nan,
            "note": "No tested coverage level satisfies both development constraints.",
        }]
    )
else:
    recommended_table = pd.DataFrame(
        [{
            "qualifying_operating_point_found": True,
            "minimum_precision": COVERAGE_CONFIG["minimum_precision"],
            "minimum_f1": COVERAGE_CONFIG["minimum_f1"],
            "recommended_coverage_pct": recommended["coverage_pct"],
            "precision": recommended["precision"],
            "recall": recommended["recall"],
            "f1": recommended["f1"],
            "manual_review_days": recommended["manual_review_days"],
        }]
    )
display(coverage_metrics)
display(recommended_table)
"""
        ),
        markdown(
            """
## 7. Write Metrics, Tables, And Figures

All headline tables are derived from the same nested outer prediction audit.
Figure 4 is the requested dual-axis view: auto-accept coverage on the horizontal
axis, days left for manual review on the left vertical axis, and stacked auto FP
and FN counts on the right vertical axis.
"""
        ),
        code(
            """
DAY_PATH = OUTPUT_DIRS["metrics"] / "01_day_metrics.csv"
INTERVAL_PATH = OUTPUT_DIRS["metrics"] / "02_interval_metrics.csv"
WINDOW_PATH = OUTPUT_DIRS["metrics"] / "03_window_iou_metrics.csv"
ENERGY_PATH = OUTPUT_DIRS["metrics"] / "04_energy_metrics.csv"
COVERAGE_PATH = OUTPUT_DIRS["metrics"] / "05_confidence_coverage_metrics.csv"
write_csv(day_metrics, DAY_PATH)
write_csv(interval_metrics, INTERVAL_PATH)
write_csv(window_metrics, WINDOW_PATH)
write_csv(energy_metrics, ENERGY_PATH)
write_csv(coverage_metrics, COVERAGE_PATH)

HEADLINE_PATH = OUTPUT_DIRS["tables"] / "table01_final_headline_metrics.csv"
SUBSTATION_PATH = OUTPUT_DIRS["tables"] / "table02_final_metrics_by_substation.csv"
LOCALISATION_PATH = OUTPUT_DIRS["tables"] / "table03_localisation_and_energy.csv"
RECOMMENDED_PATH = (
    OUTPUT_DIRS["tables"] / "table04_recommended_auto_accept_operating_point.csv"
)
headline = day_metrics.loc[
    day_metrics["aggregation"].isin(["pooled", "macro_substation"])
]
substation_table = day_metrics.loc[day_metrics["aggregation"].eq("substation")]
localisation_table = pd.concat(
    [
        window_metrics.loc[window_metrics["aggregation"].eq("pooled")].assign(
            metric_family="window_iou"
        ),
        energy_metrics.loc[energy_metrics["aggregation"].eq("pooled")].assign(
            metric_family="correction_energy"
        ),
    ],
    ignore_index=True,
    sort=False,
)
write_csv(headline, HEADLINE_PATH)
write_csv(substation_table, SUBSTATION_PATH)
write_csv(localisation_table, LOCALISATION_PATH)
write_csv(recommended_table, RECOMMENDED_PATH)

FIGURE_CONFUSION = OUTPUT_DIRS["figures"] / "fig01_final_confusion_matrices.png"
FIGURE_IOU = OUTPUT_DIRS["figures"] / "fig02_window_iou_distribution.png"
FIGURE_ENERGY = OUTPUT_DIRS["figures"] / "fig03_energy_metric_summary.png"
FIGURE_BURDEN = (
    OUTPUT_DIRS["figures"] / "fig04_auto_accept_manual_review_and_errors.png"
)
FIGURE_COVERAGE = (
    OUTPUT_DIRS["figures"] / "fig05_auto_accept_precision_recall_f1.png"
)
plot_final_confusion_matrices(day_metrics, FIGURE_CONFUSION)
plot_window_iou_distribution(window_audit, FIGURE_IOU)
plot_energy_summary(energy_metrics, FIGURE_ENERGY)
plot_auto_accept_burden(coverage_metrics, FIGURE_BURDEN)
plot_coverage_scores(coverage_metrics, FIGURE_COVERAGE)
display(FIGURE_CONFUSION)
display(FIGURE_IOU)
display(FIGURE_ENERGY)
display(FIGURE_BURDEN)
display(FIGURE_COVERAGE)
"""
        ),
        markdown(
            """
## 8. Interpretation And Limitations

Day metrics describe held-out-substation classification. Interval, window, and
energy metrics additionally test whether positive decisions target the correct
part of the curve and recover the material correction magnitude. Confidence
coverage quantifies a possible reduction in manual checking; its recommended
point is a development result, not a deployed safety guarantee.

Beta labels informed earlier feature-family development. Nested LOSO prevents
direct outer-fold leakage but does not make Beta a previously untouched external
dataset. Independent validation is still required before operational use.
"""
        ),
        code(
            """
MANIFEST_OUTPUTS = [
    DAY_PATH, INTERVAL_PATH, WINDOW_PATH, ENERGY_PATH, COVERAGE_PATH,
    HEADLINE_PATH, SUBSTATION_PATH, LOCALISATION_PATH, RECOMMENDED_PATH,
    FIGURE_CONFUSION, FIGURE_IOU, FIGURE_ENERGY, FIGURE_BURDEN, FIGURE_COVERAGE,
]
manifest = manifest_payload(
    paths=PATHS,
    config=CONFIG,
    started_at=STARTED_AT,
    inputs=[PATHS.config, PATHS.final_data / "dataset_beta.parquet", PREDICTION_PATH],
    outputs=MANIFEST_OUTPUTS,
    row_counts={
        "beta_intervals": len(interval_audit),
        "beta_substation_days": len(predictions),
        "day_metric_rows": len(day_metrics),
        "interval_metric_rows": len(interval_metrics),
        "coverage_levels": len(coverage_metrics),
    },
)
manifest["local_intermediates"] = [str(INTERVAL_AUDIT_PATH.relative_to(PATHS.article))]
MANIFEST_PATH = write_manifest(PATHS, f"{SLUG}.json", manifest)
inventory = pd.DataFrame({"path": [*MANIFEST_OUTPUTS, INTERVAL_AUDIT_PATH, MANIFEST_PATH]})
inventory["exists"] = inventory["path"].map(Path.exists)
inventory["bytes"] = inventory["path"].map(lambda path: path.stat().st_size)
display(inventory)
assert inventory["exists"].all() and inventory["bytes"].gt(0).all()
"""
        ),
    ]
    return write_notebook(notebook_dir / "02g_m9_pbm_final_evaluation.ipynb", cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        choices=["02a", "02b", "02c", "02d", "02e", "02f", "02g", "all"],
        default="all",
    )
    args = parser.parse_args()
    notebook_dir = Path(__file__).resolve().parent
    builders = {
        "02a": build_02a,
        "02b": build_02b,
        "02c": build_02c,
        "02d": build_02d,
        "02e": build_02e,
        "02f": build_02f,
        "02g": build_02g,
    }
    selected = list(builders) if args.notebook == "all" else [args.notebook]
    paths = [builders[name](notebook_dir) for name in selected]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
