<#
.SYNOPSIS
    Launches the AI Drone Swarm Command Center Streamlit Dashboard.

.DESCRIPTION
    Verifies virtual environment dependencies (Streamlit, Pandas, PyArrow) and launches
    the command center dashboard frontend on localhost.

.PARAMETER Port
    Port number for Streamlit server (default: 8501).

.PARAMETER Headless
    Run in headless mode without automatically launching system browser.

.EXAMPLE
    .\run_dashboard.ps1
    .\run_dashboard.ps1 -Port 8502 -Headless
#>
param (
    [int]$Port = 8501,
    [switch]$Headless = $false
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Get-Location
}

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment Python not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Please verify that the virtual environment is configured at 'venv/'." -ForegroundColor Yellow
    exit 1
}

Write-Host "Verifying Streamlit environment..." -ForegroundColor Cyan
& $VenvPython -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to load Streamlit from $VenvPython" -ForegroundColor Red
    exit 1
}

$AppPath = Join-Path $ProjectRoot "dashboard\app.py"
if (-not (Test-Path $AppPath)) {
    Write-Host "[ERROR] Dashboard entrypoint not found at: $AppPath" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  AI DRONE SWARM COMMAND CENTER DASHBOARD" -ForegroundColor Green
Write-Host "  URL: http://localhost:$Port" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$ArgsList = @("run", $AppPath, "--server.port", $Port)
if ($Headless) {
    $ArgsList += "--server.headless"
    $ArgsList += "true"
}

& $VenvPython -m streamlit @ArgsList

