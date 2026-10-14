#!/bin/bash
# Link the versioned hooks in tools/git-hooks/ into .git/hooks/.
# Re-run after cloning.  Does not modify git config.
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOKS_DIR"
for hook in "$REPO"/tools/git-hooks/*; do
   name="$(basename "$hook")"
   chmod +x "$hook"
   ln -sf "$hook" "$HOOKS_DIR/$name"
   echo "installed $name -> $HOOKS_DIR/$name"
done
