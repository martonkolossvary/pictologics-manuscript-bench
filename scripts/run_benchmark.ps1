param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArguments
)
$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

# The Python launcher is the single cross-platform execution contract. It is
# print-only unless the caller explicitly supplies --execute --confirm CALCULATE.
# During execution it owns and releases the Windows system-sleep assertion.
poetry run python scripts/launch_benchmark.py @LauncherArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
