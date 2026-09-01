$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python -m src.model.load --config configs/model.yaml --smoke-test --json-out outputs/stage0/model_smoke.json
}
finally {
    Pop-Location
}
