param(
    [string]$WorkspaceHost = "",
    [string]$ConfigPath = "config/databricks_connect.yaml",
    [string]$Profile = "dbx",
    [string]$DatabricksConnectVersion = "16.4.*",
    [string]$ClusterId = "0802-073102-ie1bry9e"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$cliPath = Join-Path $repoRoot ".tools\databricks-cli\databricks.exe"
$cliInstallScript = Join-Path $PSScriptRoot "install_databricks_cli.ps1"

function Get-WorkspaceHostFromYaml {
    param([string]$YamlPath)

    if (-not (Test-Path $YamlPath)) {
        return ""
    }

    foreach ($line in (Get-Content $YamlPath)) {
        if ($line -match '^\s*workspace_host\s*:\s*["'']?([^"'']+)["'']?\s*$') {
            return $matches[1].Trim()
        }
    }
    return ""
}

if ([string]::IsNullOrWhiteSpace($WorkspaceHost)) {
    $candidatePaths = @()
    if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
        $candidatePaths += $ConfigPath
    } else {
        $candidatePaths += (Join-Path $repoRoot $ConfigPath)
    }
    $candidatePaths += (Join-Path $repoRoot "config\databricks_connect.example.yaml")

    foreach ($path in $candidatePaths) {
        $foundHost = Get-WorkspaceHostFromYaml -YamlPath $path
        if (-not [string]::IsNullOrWhiteSpace($foundHost)) {
            $WorkspaceHost = $foundHost
            Write-Host "Using WorkspaceHost from config: $WorkspaceHost"
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($WorkspaceHost)) {
    throw "Workspace host not provided. Set local_connect.workspace_host in config or pass -WorkspaceHost."
}

if ($WorkspaceHost -match "<your-workspace-host>") {
    throw "Workspace host is still a placeholder. Update local_connect.workspace_host in config."
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvDir"
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment (exit code $LASTEXITCODE)."
    }
}

Write-Host "Installing Python dependencies"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in .venv (exit code $LASTEXITCODE)."
}
& $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements.txt (exit code $LASTEXITCODE)."
}

# Databricks Connect cannot coexist with pyspark packages in the same environment.
Write-Host "Removing incompatible Spark packages (if present): pyspark, pyspark-connect, pyspark-client"
& $venvPython -m pip uninstall -y pyspark pyspark-connect pyspark-client
if ($LASTEXITCODE -ne 0) {
    throw "Failed to uninstall conflicting pyspark packages (exit code $LASTEXITCODE)."
}

& $venvPython -m pip install "databricks-connect==$DatabricksConnectVersion"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install databricks-connect==$DatabricksConnectVersion."
}

if (-not (Test-Path $cliPath)) {
    Write-Host "Databricks CLI not found in repo tools. Installing..."
    & $cliInstallScript
}

if (-not (Test-Path $cliPath)) {
    throw "Databricks CLI installation failed. Expected binary at: $cliPath"
}

$env:DATABRICKS_CLI_PATH = $cliPath
Write-Host "Set DATABRICKS_CLI_PATH=$cliPath for this shell session."

Write-Host "Databricks CLI version:"
& $cliPath -v
if ($LASTEXITCODE -ne 0) {
    throw "Databricks CLI is not executable (exit code $LASTEXITCODE)."
}

Write-Host "Running OAuth login for profile '$Profile'..."
& $cliPath auth login --host $WorkspaceHost --profile $Profile
if ($LASTEXITCODE -ne 0) {
    throw "OAuth login failed for profile '$Profile' (exit code $LASTEXITCODE)."
}

$env:DATABRICKS_CONFIG_PROFILE = $Profile
Write-Host "Set DATABRICKS_CONFIG_PROFILE=$Profile for this shell session."

if ([string]::IsNullOrWhiteSpace($ClusterId)) {
    $env:DATABRICKS_SERVERLESS_COMPUTE_ID = "auto"
    Remove-Item Env:DATABRICKS_CLUSTER_ID -ErrorAction SilentlyContinue
    Write-Host "Using serverless compute for smoke test."
} else {
    $env:DATABRICKS_CLUSTER_ID = $ClusterId
    Remove-Item Env:DATABRICKS_SERVERLESS_COMPUTE_ID -ErrorAction SilentlyContinue
    Write-Host "Using cluster compute for smoke test: $ClusterId"
}

Write-Host "Current Databricks auth profiles:"
& $cliPath auth profiles
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read Databricks auth profiles (exit code $LASTEXITCODE)."
}

$smokeTest = @'
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.getOrCreate()
spark.sql("select current_user() as user, current_timestamp() as ts").show(truncate=False)
print("Databricks Connect smoke test passed.")
'@

Write-Host "Running Databricks Connect smoke test"
$smokeTest | & $venvPython -
if ($LASTEXITCODE -ne 0) {
    throw "Databricks Connect smoke test failed (exit code $LASTEXITCODE)."
}

Write-Host "Setup complete. Next:"
Write-Host "  Copy-Item config\databricks_connect.example.yaml config\databricks_connect.yaml"
Write-Host "  .\scripts\run_local_connect.ps1 -ConfigPath `"config/databricks_connect.yaml`" -Profile `"$Profile`""
