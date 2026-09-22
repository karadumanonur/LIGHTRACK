param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$packagingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $packagingDir
$distDir = Join-Path $projectDir "dist\windows"
$workDir = Join-Path $projectDir "build\pyinstaller-windows"
$specDir = Join-Path $workDir "spec"
$summaryCandidates = @(
    (Join-Path $projectDir "assets\two_period_peak_change_mean_std_pm.csv"),
    (Join-Path $projectDir "outputs\fbg1_4_5_9_10_two_periods_detrended_axle_peak_pm\two_period_peak_change_mean_std_pm.csv"),
    (Join-Path (Split-Path -Parent $projectDir) "outputs\fbg1_4_5_9_10_two_periods_detrended_axle_peak_pm\two_period_peak_change_mean_std_pm.csv")
)
$summary = $summaryCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $summary) {
    throw "Required Two Period summary CSV was not found."
}

if (-not $PythonPath) {
    $pythonCandidates = @(
        (Join-Path $projectDir ".venv\Scripts\python.exe"),
        (Join-Path $projectDir "venv\Scripts\python.exe")
    )
    $PythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $PythonPath) {
        $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    }
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Build Python was not found: $PythonPath"
}
& $PythonPath -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in the selected build environment."
}

New-Item -ItemType Directory -Path $distDir -Force | Out-Null
New-Item -ItemType Directory -Path $workDir -Force | Out-Null
New-Item -ItemType Directory -Path $specDir -Force | Out-Null
$env:PYINSTALLER_CONFIG_DIR = Join-Path $workDir "config"
New-Item -ItemType Directory -Path $env:PYINSTALLER_CONFIG_DIR -Force | Out-Null

$dataSeparator = [System.IO.Path]::PathSeparator
$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--console",
    "--name", "TrainMonitoring",
    "--distpath", $distDir,
    "--workpath", $workDir,
    "--specpath", $specDir,
    "--paths", $projectDir,
    "--collect-data", "streamlit",
    "--collect-submodules", "streamlit",
    "--copy-metadata", "streamlit",
    "--hidden-import", "vmdpy",
    "--hidden-import", "scipy._external.array_api_compat.numpy.fft",
    "--add-data", ((Join-Path $projectDir "fbg_analysis_dashboard_strict.py") + $dataSeparator + "."),
    "--add-data", ((Join-Path $projectDir ".streamlit\config.toml") + $dataSeparator + ".streamlit"),
    (Join-Path $projectDir "lightrack_exe.py")
)

$entryScript = $arguments[-1]
$arguments = $arguments[0..($arguments.Count - 2)] + @(
    "--add-data", ($summary + $dataSeparator + "assets"),
    $entryScript
)

Write-Host "Building TrainMonitoring.exe. This can take several minutes..." -ForegroundColor Cyan
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "The Train Monitoring executable build failed."
}

$appDir = Join-Path $distDir "TrainMonitoring"
Copy-Item -LiteralPath (Join-Path $packagingDir "README_EXE.md") -Destination (Join-Path $appDir "README_EXE.md") -Force
Copy-Item -LiteralPath (Join-Path $packagingDir "Choose_Train_Monitoring_Data_Folder.cmd") -Destination (Join-Path $appDir "Choose_Train_Monitoring_Data_Folder.cmd") -Force
& (Join-Path $appDir "TrainMonitoring.exe") --self-test
if ($LASTEXITCODE -ne 0) {
    throw "The packaged Train Monitoring self-test failed."
}

Write-Host "Train Monitoring Windows application created:" -ForegroundColor Green
Write-Host $appDir
