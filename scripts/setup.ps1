$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw "Poetry is required to install the benchmark controller."
}

poetry sync
poetry run python -m bench.cli env create
poetry run python -m bench.cli env verify
Write-Host "Controller and five isolated adapter environments are verified." -ForegroundColor Green
