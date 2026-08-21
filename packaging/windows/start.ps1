$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Join-Path $BundleRoot "app"
$Python = Join-Path $AppRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "小满尚未安装。请先双击 install.cmd。"
}
if (-not (Test-Path -LiteralPath (Join-Path $AppRoot "config.toml"))) {
    throw "找不到 config.toml。请先运行 install.cmd 完成迁移。"
}

Write-Host "正在启动小满：http://127.0.0.1:2236/" -ForegroundColor Cyan
Push-Location $AppRoot
try {
    & $Python main.py
}
finally {
    Pop-Location
}
