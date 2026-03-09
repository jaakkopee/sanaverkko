$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".venv"
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"

$PythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = @("py", "-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = @("python")
} else {
    Write-Error "Error: Python launcher 'py' or 'python' was not found in PATH."
    exit 1
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at: $VenvDir"
    if ($PythonCmd.Length -eq 2) {
        & $PythonCmd[0] $PythonCmd[1] -m venv $VenvDir
    } else {
        & $PythonCmd[0] -m venv $VenvDir
    }
} else {
    Write-Host "Virtual environment already exists at: $VenvDir"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Error "Error: Activation script not found at $ActivateScript"
    exit 1
}

. $ActivateScript
Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"

python -m pip install --upgrade pip

if (Test-Path $RequirementsFile) {
    Write-Host "Installing dependencies from: $RequirementsFile"
    python -m pip install -r $RequirementsFile
} else {
    Write-Warning "requirements.txt not found at $RequirementsFile"
}

Write-Host "Done."
Write-Host "If you ran this script normally, activate later with: . .\.venv\Scripts\Activate.ps1"
Write-Host "To keep activation in the current session, run: . .\setup_venv.ps1"