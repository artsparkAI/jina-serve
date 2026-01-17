#!/bin/bash
# Deploy Jina Embeddings v4 to RunPod Serverless
#
# This script pushes to GitHub. GitHub Actions then:
#   1. Builds Docker image
#   2. Pushes to GitHub Container Registry (ghcr.io)
#   3. Deploys to RunPod via API
#
# Prerequisites:
#   - gh CLI authenticated (gh auth login)
#   - RUNPOD_API_KEY added as GitHub secret (one-time setup)
#
# Get your Jina AI API key for free: https://jina.ai/?sui=apikey

set -e

REPO_NAME="jina-serve"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Jina Embeddings v4 Deployment ===${NC}"
echo ""

# Check gh is available
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is required. Install from https://cli.github.com"
    exit 1
fi

# Git setup
if [ ! -d ".git" ]; then
    echo "Initializing git..."
    git init -b main
else
    git branch -M main 2>/dev/null || true
fi

git add .
git commit -m "Jina Embeddings v4 vLLM worker" 2>/dev/null || echo "Nothing new to commit"

# GitHub setup
echo ""
if git remote get-url origin &>/dev/null; then
    echo "Pushing to existing remote..."
    git push -u origin main --force
else
    if gh repo view "$REPO_NAME" &>/dev/null; then
        echo "Adding existing repo as remote..."
        REPO_URL=$(gh repo view "$REPO_NAME" --json sshUrl -q .sshUrl)
        git remote add origin "$REPO_URL"
        git push -u origin main --force
    else
        echo "Creating new GitHub repo..."
        gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
    fi
fi

REPO_URL=$(gh repo view --json url -q .url)

# Check if RUNPOD_API_KEY secret exists
echo ""
echo -e "${YELLOW}Checking GitHub secrets...${NC}"
if ! gh secret list 2>/dev/null | grep -q "RUNPOD_API_KEY"; then
    echo ""
    echo -e "${YELLOW}RUNPOD_API_KEY secret not found.${NC}"
    echo "Please enter your RunPod API key (from https://runpod.io/console/user/settings):"
    read -s RUNPOD_KEY
    echo ""
    if [ -n "$RUNPOD_KEY" ]; then
        echo "$RUNPOD_KEY" | gh secret set RUNPOD_API_KEY
        echo -e "${GREEN}Secret added successfully.${NC}"
    else
        echo "Skipped. Add it manually: gh secret set RUNPOD_API_KEY"
    fi
else
    echo -e "${GREEN}RUNPOD_API_KEY secret exists.${NC}"
fi

# Trigger workflow
echo ""
echo -e "${YELLOW}Triggering GitHub Actions workflow...${NC}"
gh workflow run build.yml 2>/dev/null || echo "Workflow will run on push"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   DEPLOYMENT INITIATED${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Repository: $REPO_URL"
echo ""
echo "GitHub Actions will now:"
echo "  1. Build Docker image"
echo "  2. Push to ghcr.io"
echo "  3. Deploy to RunPod"
echo ""
echo -e "${YELLOW}Monitor progress:${NC}"
echo "  gh run watch"
echo ""
echo -e "${YELLOW}Or view in browser:${NC}"
echo "  $REPO_URL/actions"
echo ""
