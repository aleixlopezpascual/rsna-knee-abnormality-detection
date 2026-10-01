#!/usr/bin/env bash
# ==============================================================================
# Setup symlinks pointing all AI development assistant configs to AGENTS.md
# Supported assistants: GitHub Copilot, Anthropic Claude Code, Google Gemini CLI, Cursor
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Configuring AI assistant instruction symlinks in: $REPO_ROOT"

# Ensure .github directory exists
mkdir -p .github

# Create relative symlinks so paths remain portable across clones and operating systems
ln -sf AGENTS.md CLAUDE.md
ln -sf AGENTS.md GEMINI.md
ln -sf ../AGENTS.md .github/copilot-instructions.md

echo "✅ Symlinks successfully configured:"
ls -la CLAUDE.md GEMINI.md .github/copilot-instructions.md
