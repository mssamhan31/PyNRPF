## Release Readiness Check (2026-02-25)

Scope: pre-merge sensitive-data gate for conference release snapshot.

### 1) Automated secret scans

- Tool: `gitleaks v8.18.2` (portable binary, local run)
- History scan command:
  - `gitleaks detect --source . --log-opts="--all" --report-format json --report-path %TEMP%/gitleaks_history_all_report.json --redact --exit-code 0 --no-banner`
- Result: `0` findings.

- Worktree scan command:
  - `gitleaks detect --source . --no-git --report-format json --report-path %TEMP%/gitleaks_worktree_report.json --redact --exit-code 0 --no-banner`
- Raw result: `9` findings, all under `.venv/` (local environment files, not repo-tracked).
- Tracked-file regex sweep over `git ls-files`: `0` secret-like matches.

Conclusion: no detected secrets in tracked repository content or git history.

### 2) Large-file and artifact review

Tracked items include:

- `data/raw/rpf_dataset.parquet` (~26 MB)
- `outputs/*.pkl`, `outputs/*.json`, and publication figure/table artifacts

Status: repository currently keeps dataset and generated artifacts in git by design.

### 3) Manual checklist notes

- Potential environment identifiers found (non-secret):
  - Databricks workspace host appears in executed notebook output.
  - Example/default cluster IDs present in Databricks helper files.
- No API tokens/passwords/private keys found in tracked files.

### 4) Gate decision

- Sensitive-data gate: **PASS with caveat**.
- Caveat: environment identifiers are present; publication approved under current "publish as-is" policy.
