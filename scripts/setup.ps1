$ErrorActionPreference = "Stop"
$projectDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $projectDir

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw "Poetry is required to install the benchmark controller."
}

foreach ($version in @("3.9", "3.10", "3.12")) {
    $python = Get-Command "python$version" -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python $version is required. Install it with: py install $version"
    }
}

# uv is required on Windows because MEDimage 0.9.8 publishes the legacy
# Requires-Python specifier <=3.10, which pip interprets as <=3.10.0. uv
# correctly resolves the tested 3.10.x environment. bench.env also searches
# this per-user location when it is not exported through PATH.
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    $userBase = python -c "import site; print(site.getuserbase())"
    $uvCandidates = @(
        (Join-Path $userBase "Python312\Scripts\uv.exe"),
        (Join-Path $userBase "Scripts\uv.exe")
    )
    $uv = $uvCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $uv) {
    throw "uv is required on Windows. Install it with: python -m pip install --user 'uv>=0.8,<0.9'"
}

poetry sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run python -m bench.cli env create
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
poetry run python -m bench.cli env verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Controller and five isolated adapter environments are verified." -ForegroundColor Green
