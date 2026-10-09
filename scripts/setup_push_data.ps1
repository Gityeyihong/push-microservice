<#
.SYNOPSIS
    生成条目向量与 FAISS 索引。
.DESCRIPTION
    PUSH_USE_MOCK_VECTORS=1 时使用离线随机向量；否则使用本地 sentence-transformers 模型。
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

$catalog = if ($env:PUSH_CATALOG) { $env:PUSH_CATALOG } elseif ($env:PUSH_VIDEOS) { $env:PUSH_VIDEOS } else { "catalog.json" }
if ($env:PUSH_USE_MOCK_VECTORS -eq "1") {
    Write-Host "[push-data] MOCK 模式 — 生成随机向量 ($catalog)..."
    & $python data\generate_data.py --mock "--catalog=$catalog"
} else {
    Write-Host "[push-data] PROD 模式 — sentence-transformers ($catalog)..."
    & $python data\generate_data.py "--catalog=$catalog"
}
exit $LASTEXITCODE
