param(
    [ValidateSet('success','ambiguous','no-intent','cancel','withdraw','stale','target-shift','stale-glove')]
    [string]$Scenario = 'success',
    [switch]$Wire,
    [switch]$Tests
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$rfCandidates = @()
if ($env:ROBOFLEX_PYTHON) { $rfCandidates += $env:ROBOFLEX_PYTHON }
$rfCandidates += (Join-Path $PSScriptRoot '.venv/Scripts/python.exe')
$rfCandidates += (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
foreach ($rfName in @('py','python')) {
    $rfCommand = Get-Command $rfName -ErrorAction SilentlyContinue
    if ($rfCommand) { $rfCandidates += $rfCommand.Source }
}
$rfPython = $null
foreach ($rfCandidate in $rfCandidates) {
    if (-not (Test-Path -LiteralPath $rfCandidate)) { continue }
    try {
        & $rfCandidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $rfPython = $rfCandidate; break }
    } catch { }
}
if (-not $rfPython) { throw 'Python 3.10+ was not found. Install Python or set ROBOFLEX_PYTHON.' }
if ($Tests) {
    & $rfPython -m unittest discover -s tests -v
} else {
    $rfArgs = @('-m','roboflex','demo','--scenario',$Scenario)
    if ($Wire) { $rfArgs += '--wire' }
    & $rfPython @rfArgs
}
exit $LASTEXITCODE
