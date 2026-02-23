param(
    [string]$Version = "latest",
    [string]$InstallDir = ".tools/databricks-cli"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required but not found on PATH."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([System.IO.Path]::IsPathRooted($InstallDir)) {
    $resolvedInstallDir = $InstallDir
} else {
    $resolvedInstallDir = Join-Path $repoRoot $InstallDir
}

if ($Version -eq "latest") {
    $release = curl.exe -sL "https://api.github.com/repos/databricks/cli/releases/latest" | ConvertFrom-Json
    $Version = $release.tag_name.TrimStart("v")
}

$assetName = "databricks_cli_${Version}_windows_amd64.zip"
$downloadUrl = "https://github.com/databricks/cli/releases/download/v${Version}/${assetName}"
$zipPath = Join-Path $resolvedInstallDir $assetName
$cliPath = Join-Path $resolvedInstallDir "databricks.exe"

New-Item -ItemType Directory -Path $resolvedInstallDir -Force | Out-Null

Write-Host "Downloading Databricks CLI $Version from $downloadUrl"
curl.exe -L $downloadUrl -o $zipPath
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Download failed with certificate revocation check. Retrying with --ssl-no-revoke."
    curl.exe --ssl-no-revoke -L $downloadUrl -o $zipPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download Databricks CLI from $downloadUrl"
    }
}

Expand-Archive -Path $zipPath -DestinationPath $resolvedInstallDir -Force

if (-not (Test-Path $cliPath)) {
    throw "Databricks CLI binary not found after extraction: $cliPath"
}

Write-Host "Installed Databricks CLI to: $resolvedInstallDir"
& $cliPath -v
Write-Host "Use this binary directly:"
Write-Host "  $cliPath"
