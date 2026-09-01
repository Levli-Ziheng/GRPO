$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python -m src.utils.check_env --strict --json-out outputs/stage0/environment.json
    & $Python -m src.utils.memory_estimator --config configs/model.yaml --json-out outputs/stage0/memory_estimate.json
}
finally {
    Pop-Location
}
