<#
.SYNOPSIS
    启动 Push Microservice（开发/本机）。
.DESCRIPTION
    优先使用 .venv；缺失向量索引时自动生成（尊重 PUSH_USE_MOCK_VECTORS）。
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

if (-not (Test-Path "data\faiss_index.bin")) {
    Write-Host "[push] 缺少向量索引，生成中..."
    & (Join-Path $PSScriptRoot "setup_push_data.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$port = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8000" }
Write-Host "[push] 启动 http://0.0.0.0:$port ..."
& $python -m uvicorn server.main:app --host 0.0.0.0 --port $port
