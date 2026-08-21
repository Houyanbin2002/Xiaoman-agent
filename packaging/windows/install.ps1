$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Join-Path $BundleRoot "app"
$MigrationRoot = Join-Path $BundleRoot "migration"
$ManifestPath = Join-Path $BundleRoot "manifest.json"

if (-not (Test-Path -LiteralPath (Join-Path $AppRoot "main.py"))) {
    throw "迁移包不完整：找不到 app\main.py。请先完整解压 ZIP。"
}

function Find-Python312 {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Command = $py.Source; Prefix = @("-3.12") }
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Command = $python.Source; Prefix = @() }
        }
    }
    $defaultPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $defaultPython) {
        return @{ Command = $defaultPython; Prefix = @() }
    }
    return $null
}

$python = Find-Python312
if (-not $python) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "没有找到 Python 3.12，也没有 winget。请先从 python.org 安装 Python 3.12 x64，再重新运行 install.cmd。"
    }
    Write-Host "正在安装 Python 3.12..." -ForegroundColor Cyan
    & $winget.Source install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 安装失败，退出码 $LASTEXITCODE。"
    }
    $python = Find-Python312
    if (-not $python) {
        throw "Python 已安装但当前终端仍未找到，请关闭窗口后重新运行 install.cmd。"
    }
}

$PythonCommand = [string]$python.Command
$PythonPrefix = @($python.Prefix)

$VenvPython = Join-Path $AppRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "正在创建小满运行环境..." -ForegroundColor Cyan
    & $PythonCommand @PythonPrefix -m venv (Join-Path $AppRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "创建 Python 虚拟环境失败。" }
}

Write-Host "正在安装 Python 依赖，这一步需要联网..." -ForegroundColor Cyan
& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $AppRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Python 依赖安装失败。" }

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
function Restore-PrivateDirectory([string]$Source, [string]$Target) {
    if (-not (Test-Path -LiteralPath $Source)) { return }
    $Prepared = "$Target.incoming-$Timestamp"
    if (Test-Path -LiteralPath $Prepared) {
        Remove-Item -LiteralPath $Prepared -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Prepared -Recurse -Force
    if (Test-Path -LiteralPath $Target) {
        $Backup = "$Target.before-migration-$Timestamp"
        Move-Item -LiteralPath $Target -Destination $Backup
        Write-Host "已备份原目录：$Backup" -ForegroundColor Yellow
    }
    Move-Item -LiteralPath $Prepared -Destination $Target
}

$XiaomanHome = Join-Path $HOME ".xiaoman"
$PluginHome = Join-Path $HOME ".xiaoman-plugin"
Restore-PrivateDirectory (Join-Path $MigrationRoot "xiaoman") $XiaomanHome
Restore-PrivateDirectory (Join-Path $MigrationRoot "xiaoman-plugin") $PluginHome

$ConfigSource = Join-Path $MigrationRoot "config.toml"
if (Test-Path -LiteralPath $ConfigSource) {
    Copy-Item -LiteralPath $ConfigSource -Destination (Join-Path $AppRoot "config.toml") -Force
}
$OverrideRoot = Join-Path $MigrationRoot "app-overrides"
if (Test-Path -LiteralPath $OverrideRoot) {
    Get-ChildItem -LiteralPath $OverrideRoot -Recurse -File | ForEach-Object {
        $Relative = $_.FullName.Substring($OverrideRoot.Length).TrimStart([char[]]"\/")
        $Target = Join-Path $AppRoot $Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Target -Force
    }
}

if (Test-Path -LiteralPath $ManifestPath) {
    $Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    $OldHome = [string]$Manifest.source_user_home
    if ($OldHome) {
        $TextFiles = @(
            (Join-Path $AppRoot "config.toml"),
            (Join-Path $XiaomanHome "workspace\mcp_servers.json"),
            (Join-Path $XiaomanHome "workspace\proactive_sources.json"),
            (Join-Path $PluginHome "registry.json")
        )
        foreach ($TextFile in $TextFiles) {
            if (-not (Test-Path -LiteralPath $TextFile)) { continue }
            $Content = Get-Content -Raw -LiteralPath $TextFile
            $Content = $Content.Replace($OldHome, $HOME)
            $Content = $Content.Replace($OldHome.Replace("\", "/"), $HOME.Replace("\", "/"))
            Set-Content -LiteralPath $TextFile -Value $Content -Encoding utf8
        }
    }
}

$McpConfig = Join-Path $XiaomanHome "workspace\mcp_servers.json"
if ((Test-Path -LiteralPath $McpConfig) -and (Select-String -LiteralPath $McpConfig -Pattern '"markitdown"' -Quiet)) {
    Write-Host "正在重建文档解析 MCP 运行环境..." -ForegroundColor Cyan
    Push-Location $AppRoot
    try {
        & $VenvPython -c "from agent.mcp.catalog import _ensure_markitdown_runtime; print(_ensure_markitdown_runtime())"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "MarkItDown MCP 重建失败，可稍后在“小满 → 设置与扩展 → MCP”中重新安装。"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "小满安装与数据迁移完成。" -ForegroundColor Green
Write-Host "请运行 start.cmd 启动，浏览器访问 http://127.0.0.1:2236/。"
Write-Host "Notion OAuth、微信扫码等系统凭据无法跨电脑复制，需要在界面中重新授权。" -ForegroundColor Yellow
