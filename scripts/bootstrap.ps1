$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

& $Python -m pip install --upgrade pip setuptools wheel
$TorchVersion = if ($env:TORCH_VERSION) { $env:TORCH_VERSION } else { "2.8.0" }
$TorchIndexUrl = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }
& $Python -m pip install "torch==$TorchVersion" --index-url $TorchIndexUrl
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

& $Python -c "import sys; print('Dependencies installed into active environment:', sys.executable)"
