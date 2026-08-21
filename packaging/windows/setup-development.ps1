$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Join-Path $BundleRoot "app"
$InstallScript = Join-Path $BundleRoot "install.ps1"
$VenvPython = Join-Path $AppRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "尚未建立运行环境，先执行基础安装..." -ForegroundColor Cyan
    & $InstallScript
}

function Ensure-WingetPackage(
    [string]$CommandName,
    [string]$PackageId,
    [string]$DisplayName
) {
    $Command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "缺少 $DisplayName 和 winget，请手动安装 $DisplayName 后重新运行。"
    }
    Write-Host "正在安装 $DisplayName..." -ForegroundColor Cyan
    & $Winget.Source install --id $PackageId -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "$DisplayName 安装失败，退出码 $LASTEXITCODE。" }
    return $null
}

$Git = Ensure-WingetPackage "git" "Git.Git" "Git"
if (-not $Git) {
    $GitCandidate = Join-Path $env:ProgramFiles "Git\cmd\git.exe"
    if (Test-Path -LiteralPath $GitCandidate) { $Git = $GitCandidate }
}

$Npm = Ensure-WingetPackage "npm.cmd" "OpenJS.NodeJS.LTS" "Node.js LTS"
if (-not $Npm) {
    $NpmCandidate = Join-Path $env:ProgramFiles "nodejs\npm.cmd"
    if (Test-Path -LiteralPath $NpmCandidate) { $Npm = $NpmCandidate }
}
if (-not $Npm) {
    throw "Node.js 已安装但当前终端未找到 npm。请关闭窗口后重新运行 setup-development.cmd。"
}

$DevRequirements = Join-Path $AppRoot "requirements-dev.txt"
if (Test-Path -LiteralPath $DevRequirements) {
    Write-Host "正在安装 Python 开发与测试依赖..." -ForegroundColor Cyan
    & $VenvPython -m pip install --disable-pip-version-check -r $DevRequirements
    if ($LASTEXITCODE -ne 0) { throw "Python 开发依赖安装失败。" }
}

Write-Host "正在安装前端开发依赖..." -ForegroundColor Cyan
Push-Location $AppRoot
try {
    & $Npm ci
    if ($LASTEXITCODE -ne 0) { throw "前端依赖安装失败。" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "小满开发环境准备完成。" -ForegroundColor Green
Write-Host "源码目录：$AppRoot"
Write-Host "后端启动：双击 start.cmd"
Write-Host "前端开发：在 app 目录执行 npm run dev"
Write-Host "Git 分支和未提交修改已保留；首次推送前请在新电脑重新登录 GitHub。" -ForegroundColor Yellow
