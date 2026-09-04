param(
    [Parameter(Mandatory = $true)]
    [string]$ResultRoot,
    [string]$HostProfile = "configs/benchmark/hosts/windows-9800x3d-01.json"
)
$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

python scripts/audit_repository_contents.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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
    --host-profile $HostProfile `
    --validate-plans
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
