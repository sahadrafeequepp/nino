#!/usr/bin/env bash
# ==============================================================================
# Safe Push Helper for Linux / macOS / Git Bash
# Usage:
#     ./scripts/safe_push.sh "Your commit message here"
# ==============================================================================

set -e

COMMIT_MSG="$1"

echo ""
echo "🛡️  [Nino Safe Push] Starting automated security scan..."

# Find Python
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Run Pre-Stage Scan
echo "🔍 Verifying repository security..."
$PYTHON_BIN scripts/check_secrets.py --all

# Prompt for message if empty
if [ -z "$COMMIT_MSG" ]; then
    read -rp "📝 Enter commit message: " COMMIT_MSG
fi

if [ -z "$COMMIT_MSG" ]; then
    echo "❌ Commit message cannot be empty."
    exit 1
fi

# Stage changes safely
echo "📦 Staging changes..."
git add .

# Scan Staged
echo "🔍 Scanning staged changes..."
$PYTHON_BIN scripts/check_secrets.py --staged

# Commit
echo "💾 Committing: '$COMMIT_MSG'..."
git commit -m "$COMMIT_MSG"

# Push if remote exists
if git remote | grep -q "origin"; then
    BRANCH=$(git branch --show-current)
    echo "🚀 Pushing to origin/$BRANCH..."
    git push origin "$BRANCH"
    echo ""
    echo "✨ Successfully upgraded and pushed safely to GitHub!"
else
    echo ""
    echo "ℹ️  Committed locally. (No remote configured yet. Add remote with: git remote add origin <url>)"
fi

