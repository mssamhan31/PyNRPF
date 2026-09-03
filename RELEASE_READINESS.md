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

- `publication/1_conference_paper/dataset/raw/rpf_dataset.parquet`
- `publication/2_journal_article/dataset/` (final, processed and oracle datasets)
- `publication/1_conference_paper/outputs/*.json`, `publication/1_conference_paper/outputs/*.yaml`
- publication figure/table exports under `publication/1_conference_paper/outputs/publication_figures` and `publication/1_conference_paper/outputs/publication_tables`

Status: repository intentionally versions dataset and generated outputs.

### 4) Gate decision

- Sensitive-data gate: **PASS**.
- No active secrets detected in tracked content or git history.
- No concrete environment identifiers detected in tracked content.

---

## Follow-up review (2026-09-03)

The checks above were run on 2026-02-25 and describe the tree as it stood then;
paths in section 3 have been corrected to their current locations. A later audit
(`.ai/review-20260902.md`) found no secrets, but did find absolute Windows paths
and machine account names embedded in committed notebook outputs and generated
manifests, which the 2026-02-25 pattern sweep did not cover. Those have since
been removed and the code that wrote them now emits repository-relative paths.

Re-run the gate before the next release rather than relying on this record.
