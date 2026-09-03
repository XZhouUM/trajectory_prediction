#!/bin/bash
set -e

echo "Setting up Git hooks..."

git config core.hooksPath .githooks

if [ -f .githooks/pre-push ]; then
  chmod +x .githooks/pre-push
  echo "pre-push hook is installed and executable."
else
  echo "Warning: .githooks/pre-push not found!"
fi
