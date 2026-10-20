#!/bin/bash
# Usage: bash setup_colab.sh <repo_url> <git_name> <git_email>
set -e
REPO_URL="$1"
GIT_NAME="$2"
GIT_EMAIL="$3"
REPO_DIR="/content/finrag-robustness"
BRANCH="person-b/dev"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"
echo "Ready: $(git branch --show-current) in $REPO_DIR"
