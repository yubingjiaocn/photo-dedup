# Photo Dedup - local review trial launcher (review-only; never moves/deletes/edits photos)
#
#   .\start-local-review.ps1 -Root "E:\照片\2026-10-01" -Output "D:\photo-review\2026-10-01"
#   .\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Reopen
#   .\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Export
#
# Optional VLM suggestions (opt-in; key only from the PHOTO_VLM_API_KEY env var):
#   $env:PHOTO_VLM_API_KEY = "<your key, if your gateway needs one>"
#   .\start-local-review.ps1 -Root "E:\照片\2026-10-01" -Output "D:\photo-review\2026-10-01" -Vlm `
#       -VlmBaseUrl "http://127.0.0.1:4000/v1" -VlmModel "<your-model-alias>"
#   .\start-local-review.ps1 -Output "D:\photo-review\2026-10-01" -Judge -VlmBaseUrl ... -VlmModel ...
#
# Requires .\setup_windows.ps1 to have created .venv first.

[CmdletBinding()]
param(
    [string]$Root,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$Limit = 100,
    [switch]$Reopen,
    [switch]$Export,
    [switch]$Rescan,
    [switch]$SmokeStub,
    [int]$Port = 0,
    [switch]$NoOpen,
    [switch]$Vlm,
    [switch]$Judge,
    [string]$VlmBaseUrl,
    [string]$VlmModel,
    [int]$VlmMaxGroups = 8,
    [double]$VlmTimeout = 120,
    [switch]$VlmNoResponseFormat,
    [switch]$VlmForce
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "[ERROR] .venv not found. Run .\setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$VlmArgs = @("--vlm-max-groups", "$VlmMaxGroups",
             "--vlm-timeout", $VlmTimeout.ToString([System.Globalization.CultureInfo]::InvariantCulture))
if ($VlmBaseUrl) { $VlmArgs += @("--vlm-base-url", $VlmBaseUrl) }
if ($VlmModel) { $VlmArgs += @("--vlm-model", $VlmModel) }
if ($VlmNoResponseFormat) { $VlmArgs += "--vlm-no-response-format" }
if ($VlmForce) { $VlmArgs += "--vlm-force" }

if ($Export) {
    $PyArgs = @("-m", "src.local_trial", "export", "--output", $Output)
}
elseif ($Judge) {
    $PyArgs = @("-m", "src.local_trial", "judge", "--output", $Output) + $VlmArgs
}
elseif ($Reopen) {
    $PyArgs = @("-m", "src.local_trial", "reopen", "--output", $Output, "--port", "$Port")
}
else {
    if (-not $Root) {
        Write-Host "[ERROR] -Root is required for a new review (or use -Reopen / -Export)." -ForegroundColor Red
        exit 2
    }
    $PyArgs = @("-m", "src.local_trial", "start", "--root", $Root, "--output", $Output,
                "--limit", "$Limit", "--port", "$Port")
    if ($Rescan) { $PyArgs += "--rescan" }
    if ($SmokeStub) { $PyArgs += "--smoke-stub" }
    if ($Vlm) { $PyArgs += @("--vlm") + $VlmArgs }
}
if ($NoOpen -and -not $Export -and -not $Judge) { $PyArgs += "--no-open" }

Push-Location -LiteralPath $PSScriptRoot
try {
    # Each argument is passed as its own token, so spaces and Chinese characters
    # in -Root / -Output survive without extra quoting.
    & $VenvPython @PyArgs
    $Code = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($Code -ne 0) {
    Write-Host "[ERROR] exit code $Code. Send the console text above and $Output\photo-dedup.log" -ForegroundColor Red
}
else {
    Write-Host "Done. Originals were only read. Reopen later with: .\start-local-review.ps1 -Output `"$Output`" -Reopen"
}
exit $Code
