#!/bin/bash
# PostToolUse hook: Auto-update data ontology when data files or ETL scripts change.
#
# Triggers on:
#   - Bash commands that touch data/ or site_scoring/data_transform.py
#   - Write/Edit operations on data files, ETL scripts, or config.py
#
# Runs: python scripts/generate_data_ontology.py --quick
# Output: docs/data_ontology.yaml (regenerated)
#
# Exit codes: 0 = allow (always allows the action, ontology update is a side effect)

set -eo pipefail

# Resolve project dir: use CLAUDE_PROJECT_DIR if set, otherwise script's grandparent
PROJ_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

INPUT=$(cat)

# Extract tool info from stdin JSON
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input // empty')

# Patterns that should trigger ontology regeneration
DATA_PATTERNS=(
    "data/input/"
    "data/processed/"
    "site_scoring/data_transform"
    "site_scoring/config.py"
    "site_scoring/data_loader"
    "site_scoring/data/"
    "scripts/generate_data_ontology"
)

should_update=false

case "$TOOL_NAME" in
    Bash)
        COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
        for pattern in "${DATA_PATTERNS[@]}"; do
            if echo "$COMMAND" | grep -q "$pattern"; then
                should_update=true
                break
            fi
        done
        ;;
    Write|Edit)
        FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
        for pattern in "${DATA_PATTERNS[@]}"; do
            if echo "$FILE_PATH" | grep -q "$pattern"; then
                should_update=true
                break
            fi
        done
        ;;
esac

if [ "$should_update" = true ]; then
    echo "Updating data ontology..." >&2

    # Run in quick mode to avoid blocking on large file reads
    cd "$PROJ_DIR"
    if python3 scripts/generate_data_ontology.py --quick 2>&1 | tail -5 >&2; then
        echo "Data ontology updated: docs/data_ontology.yaml" >&2
    else
        echo "Warning: Data ontology update failed (non-blocking)" >&2
    fi
fi

# Always allow the action to proceed
exit 0
