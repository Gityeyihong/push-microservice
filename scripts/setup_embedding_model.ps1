<#
.SYNOPSIS
    下载本地 embedding 模型到 models/（离线推理用）。
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    if (Get-Command python -ErrorAction SilentlyContinue) { $python = "python" }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $python = "py" }
    else { throw "[ERROR] 未找到 Python。" }
}

Write-Host "[setup] 下载 embedding 模型到 models/ ..."
& $python scripts\download_embedding_model.py --method hub @args
exit $LASTEXITCODE
