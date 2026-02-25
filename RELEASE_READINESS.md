## Release Readiness Check (2026-02-25)

Scope: pre-release sensitive-data gate for conference citation snapshot.

### 1) Automated secret scans

- Tool: `gitleaks v8.30.0` (portable binary, local run)
- Git history scan command:
  - `gitleaks detect --source . --log-opts="--all" --report-format json --report-path .tmp_gitleaks_history_all_report.json --redact --exit-code 0 --no-banner`
- Result: `0` findings.

- Tracked-content snapshot scan:
  - `git archive --format=tar -o .tmp_snapshot_<timestamp>.tar HEAD`
  - `tar -xf .tmp_snapshot_<timestamp>.tar -C .tmp_snapshot_<timestamp>`
  - `gitleaks detect --source .tmp_snapshot_<timestamp> --no-git --report-format json --report-path .tmp_gitleaks_tracked_snapshot_report.json --redact --exit-code 0 --no-banner`
- Result: `0` findings on tracked files.

Notes:
- A full local worktree scan including `.venv/` reports findings in third-party package test/dev files, but these are untracked and outside release scope.

### 2) Pattern scans over tracked files

- Secret-signature regex sweep over `git ls-files`:
  - AWS/GitHub/Slack/Google key signatures and private-key headers
  - Result: `0` hits
- Concrete environment identifier sweep:
  - Databricks workspace/cluster-style identifiers and workspace host URL patterns
  - Result: `0` hits

### 3) Large-file and artifact review

Tracked artifacts include:

- `data/raw/rpf_dataset.parquet`
- `outputs/*.json`, `outputs/*.yaml`, `outputs/*.pkl`
- publication figure/table exports under `outputs/publication_figures` and `outputs/publication_tables`

Status: repository intentionally versions dataset and generated outputs.

### 4) Gate decision

- Sensitive-data gate: **PASS**.
- No active secrets detected in tracked content or git history.
- No concrete environment identifiers detected in tracked content.
