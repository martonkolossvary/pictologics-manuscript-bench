$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw "Poetry is required to install the benchmark controller."
}

poetry sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# bench.env resolves interpreters via uv, the Windows py launcher, or PATH.
# It also checks MEDimage's required uv >=0.8,<0.9 before creating environments,
# including per-user uv installations that are not exported through PATH.
poetry run python -m bench.cli env create
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run python -m bench.cli env verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Controller and five isolated adapter environments are verified." -ForegroundColor Green
