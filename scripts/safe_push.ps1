# ==============================================================================
# Safe Push Helper for Windows PowerShell
# Usage:
#     .\scripts\safe_push.ps1 "Your commit message here"
# ==============================================================================

param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$CommitMessage = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "`n[Nino Safe Push] Starting automated security scan..." -ForegroundColor Cyan

# 1. Locate Python executable
$PythonBin = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PythonBin = ".venv\Scripts\python.exe"
}

# 2. Run Pre-Stage Security Scan
Write-Host "[Scan] Verifying repository security..." -ForegroundColor Gray
& $PythonBin scripts\check_secrets.py --all
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ABORTED] Security scan detected sensitive secrets or files! Commit aborted." -ForegroundColor Red
    exit 1
}

# 3. Prompt for commit message if not provided
if (-not $CommitMessage -or $CommitMessage.Trim() -eq "") {
    $CommitMessage = Read-Host "`nEnter commit message"
}

if (-not $CommitMessage -or $CommitMessage.Trim() -eq "") {
    Write-Host "[ERROR] Commit message cannot be empty." -ForegroundColor Red
    exit 1
}

# 4. Stage changes safely (guarded by .gitignore)
Write-Host "[Git] Staging changes..." -ForegroundColor Gray
git add .

# 5. Verify staged changes with the guard
Write-Host "[Scan] Scanning staged changes..." -ForegroundColor Gray
& $PythonBin scripts\check_secrets.py --staged
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ABORTED] Staged files contain sensitive information! Unstaging..." -ForegroundColor Red
    git reset HEAD
    exit 1
}

# 6. Commit
Write-Host "[Git] Committing: '$CommitMessage'..." -ForegroundColor Green
git commit -m "$CommitMessage"

# 7. Ask before pushing if remote is configured
$remotes = git remote
if ($remotes) {
    $currentBranch = (git branch --show-current).Trim()
    Write-Host "[Git] Pushing to origin/$currentBranch..." -ForegroundColor Cyan
    git push origin $currentBranch
    Write-Host "`n[SUCCESS] Successfully upgraded and pushed safely to GitHub!" -ForegroundColor Green
} else {
    Write-Host "`n[INFO] Committed locally. (No remote configured yet. Add remote with: git remote add origin <url>)" -ForegroundColor Yellow
}
