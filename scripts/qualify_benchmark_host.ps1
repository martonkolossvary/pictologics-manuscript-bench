param(
    [Parameter(Mandatory = $true)]
    [string]$ResultRoot,
    [Parameter(Mandatory = $true)]
    [string]$MachineId,
    [string]$MachineLabel = $MachineId
)
$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

poetry check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run python -m bench.cli env verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run python scripts/launch_benchmark.py `
    --workspace-root data/benchmark `
    --result-root $ResultRoot `
    --machine-id $MachineId `
    --machine-label $MachineLabel `
    --validate-plans
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
