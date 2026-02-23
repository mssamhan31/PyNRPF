param(
    [string]$ConfigPath = "config/databricks_connect.yaml",
    [string]$Profile = "dbx",
    [string]$ClusterId
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$cliPath = Join-Path $repoRoot ".tools\databricks-cli\databricks.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup_local_connect.ps1 first."
}

if (-not (Test-Path $cliPath)) {
    throw "Databricks CLI not found at $cliPath. Run .\scripts\install_databricks_cli.ps1 first."
}

if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $resolvedConfigPath = $ConfigPath
    $runConfigPath = $ConfigPath
} else {
    $resolvedConfigPath = Join-Path $repoRoot $ConfigPath
    $runConfigPath = $ConfigPath
}

if (-not (Test-Path $resolvedConfigPath)) {
    throw "Config file not found: $resolvedConfigPath"
}

$env:DATABRICKS_CLI_PATH = $cliPath
$env:DATABRICKS_CONFIG_PROFILE = $Profile
Write-Host "Using DATABRICKS_CONFIG_PROFILE=$Profile"

if ([string]::IsNullOrWhiteSpace($ClusterId)) {
    Remove-Item Env:DATABRICKS_CLUSTER_ID -ErrorAction SilentlyContinue
    Remove-Item Env:DATABRICKS_SERVERLESS_COMPUTE_ID -ErrorAction SilentlyContinue
    Write-Host "Using compute settings from $runConfigPath"
} else {
    $env:DATABRICKS_CLUSTER_ID = $ClusterId
    Remove-Item Env:DATABRICKS_SERVERLESS_COMPUTE_ID -ErrorAction SilentlyContinue
    Write-Host "Using cluster override (DATABRICKS_CLUSTER_ID=$ClusterId)"
}

Write-Host "Running Databricks hello check with config: $runConfigPath"

Push-Location $repoRoot
try {
    & $venvPython -m src.hello_databricks --config $runConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Local Databricks Connect run failed (exit code $LASTEXITCODE)."
    }
} finally {
    Pop-Location
}
