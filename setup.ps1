# setup.ps1
# PowerShell setup script for the SanaVerkko project on Windows.
# This script:
#   1. Verifies that Python 3 is available on the system.
#   2. Creates a Python virtual environment (.venv) in the repository root.
#   3. Activates the virtual environment.
#   4. Upgrades pip to the latest version.
#   5. Installs all required dependencies listed in requirements.txt.
#
# Usage: Right-click this file and choose "Run with PowerShell", or execute
#   it from a PowerShell prompt:
#       .\setup.ps1
#
# If you receive an execution-policy error, run the following command first
# (once, in an elevated PowerShell window) to allow local scripts:
#       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# ---------------------------------------------------------------------------
# Helper: write a coloured message to the host.
# Parameters:
#   $Message  - The text to display.
#   $Color    - A ConsoleColor value (default: Cyan).
# ---------------------------------------------------------------------------
function Write-Step {
    param(
        [string]$Message,
        [System.ConsoleColor]$Color = [System.ConsoleColor]::Cyan
    )
    Write-Host "`n$Message" -ForegroundColor $Color
}

# ---------------------------------------------------------------------------
# Helper: write an error message and exit with a non-zero code.
# Parameters:
#   $Message  - The error description shown to the user.
#   $ExitCode - The process exit code (default: 1).
# ---------------------------------------------------------------------------
function Exit-WithError {
    param(
        [string]$Message,
        [int]$ExitCode = 1
    )
    Write-Host "`n[ERROR] $Message" -ForegroundColor Red
    exit $ExitCode
}

# ===========================================================================
# Determine the directory that contains this script so that relative paths
# (requirements.txt, .venv) resolve correctly regardless of where the user
# invokes the script from.
# ===========================================================================
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ===========================================================================
# STEP 1 – Verify Python availability
# ===========================================================================
Write-Step "Step 1/5 – Checking for Python 3..."

# Try 'python' first (common on Windows), then 'python3' as a fallback.
$pythonCmd = $null
foreach ($candidate in @("python", "python3")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        # Confirm the found executable is actually Python 3.
        $versionOutput = & $candidate --version 2>&1
        if ($versionOutput -match "Python 3") {
            $pythonCmd = $candidate
            break
        }
    }
}

if (-not $pythonCmd) {
    Exit-WithError (
        "Python 3 was not found on this system.`n" +
        "  Please install Python 3.11 or newer from https://www.python.org/downloads/ " +
        "and ensure it is added to your PATH."
    )
}

# Display the detected Python version for the user's information.
$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "  Found: $pythonVersion  (command: $pythonCmd)" -ForegroundColor Green

# ===========================================================================
# STEP 2 – Create the virtual environment
# ===========================================================================
Write-Step "Step 2/5 – Creating virtual environment in '.venv'..."

$venvPath = Join-Path $scriptDir ".venv"

# Only create a fresh virtual environment when one does not already exist,
# which avoids overwriting an existing environment and speeds up re-runs.
if (Test-Path $venvPath) {
    Write-Host "  Virtual environment already exists at '$venvPath'. Skipping creation." -ForegroundColor Yellow
} else {
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Failed to create the virtual environment at '$venvPath'."
    }
    Write-Host "  Virtual environment created successfully." -ForegroundColor Green
}

# ===========================================================================
# STEP 3 – Activate the virtual environment
# ===========================================================================
Write-Step "Step 3/5 – Activating virtual environment..."

# The activation script for PowerShell is located at:
#   .venv\Scripts\Activate.ps1
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Exit-WithError (
        "Activation script not found at '$activateScript'.`n" +
        "  The virtual environment may be corrupted. " +
        "Delete the '.venv' folder and run this script again."
    )
}

# Dot-source the activation script so that the environment changes apply to
# the current PowerShell session (PATH, VIRTUAL_ENV, etc.).
. $activateScript
Write-Host "  Virtual environment activated." -ForegroundColor Green

# ===========================================================================
# STEP 4 – Upgrade pip
# ===========================================================================
Write-Step "Step 4/5 – Upgrading pip..."

# Upgrading pip before installing packages reduces the chance of dependency-
# resolution issues caused by an outdated pip version.
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Exit-WithError "Failed to upgrade pip."
}
Write-Host "  pip upgraded successfully." -ForegroundColor Green

# ===========================================================================
# STEP 5 – Install project dependencies
# ===========================================================================
Write-Step "Step 5/5 – Installing dependencies from 'requirements.txt'..."

$requirementsFile = Join-Path $scriptDir "requirements.txt"

if (-not (Test-Path $requirementsFile)) {
    Exit-WithError (
        "'requirements.txt' not found at '$requirementsFile'.`n" +
        "  Please ensure you are running this script from the repository root."
    )
}

# Install all packages listed in requirements.txt into the active venv.
python -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    Exit-WithError (
        "Dependency installation failed.`n" +
        "  Check the output above for details and ensure you have an active " +
        "internet connection."
    )
}

# ===========================================================================
# Done
# ===========================================================================
Write-Step "Setup complete!" -Color Green
Write-Host @"

  The SanaVerkko project is ready to use.

  To run the application, make sure the virtual environment is active and
  then execute:

      python sanaVerkkoCore.py

  If you open a new PowerShell window, re-activate the environment first:

      .\.venv\Scripts\Activate.ps1

"@ -ForegroundColor Green
