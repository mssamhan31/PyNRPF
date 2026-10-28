# PyNRPF

PyNRPF (Python for Network Reverse Power Flow) detects and corrects a wrong reverse power flow
(RPF) sign in fifteen-minute substation net-load data.

When rooftop solar exceeds local demand a substation exports, and net load is genuinely
negative. Some meters store that export as an import: the recorded trace dips towards zero
instead of crossing it. The reading passes ordinary validation but is wrong by twice its
value, and it corrupts every minimum-demand study and net-load forecast built on it.

The method, M9, asks one physical question of every day: for which daytime window would
flipping the recorded sign make the implied demand more plausible, how strong is that
evidence, and is it strong enough to act on? It needs no training, gives every day a
calibrated probability, and applies one control: correct automatically, keep automatically,
or send the day to a person.

## Three ways to run it

```python
import pynrpf
result = pynrpf.run(frame)                 # a pandas frame of readings
result.site_days, result.intervals
```

```python
from pynrpf.spark import run_per_site      # a Spark frame, one site at a time
site_days, intervals = run_per_site(sdf)
```

```powershell
pynrpf run readings.csv --out results/     # a CSV, no code
```

## Where things are

| page | what it covers |
|---|---|
| [Install](install.md) | pip, the extras, Databricks |
| [Quick start](quickstart.md) | one call, and the two output tables column by column |
| [Spark](spark.md) | the per-site adapter and what the calling pipeline owns |
| [Command line](cli.md) | `pynrpf run` |
| [API](api.md) | every public name |
| [Method](method.md) | the nine steps of M9, with the formulas and a worked day |
| [Calibration](calibration.md) | the release numbers, and refitting for a new population |
| [Overrides](overrides.md) | the review-table schema for human decisions |
| [Data](data.md) | the Alpha, Beta and Gamma datasets and how they were labelled |
| [Evaluation](evaluation.md) | how the paper judges a method |
| [Reproduce the paper](reproduce.md) | the notebooks that regenerate every figure and table |
| [Development](development.md) | tests, lint, continuous integration, releases |

The journal paper's code, data and results live under `publication/2_journal_article/`;
the conference paper's archive under `publication/1_conference_paper/`.
