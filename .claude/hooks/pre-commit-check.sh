#!/bin/bash
# Pre-commit hook: Runs pytest before allowing git commits
# Exit codes: 0 = allow, 2 = block with message

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only intercept git commit commands
if echo "$COMMAND" | grep -qE '^git commit'; then
  echo "🧪 Running pytest before commit..." >&2

  # Run pytest (skip slow/integration tests and training tests that require live model execution)
  if ! python3 -m pytest tests/test_api_sites.py tests/test_regression.py tests/test_revenue_consistency.py --ignore=tests/slow --tb=short -q 2>&1; then
    echo "" >&2
    echo "❌ Tests failed! Fix the errors before committing." >&2
    echo "   Run 'python3 -m pytest -v' to see detailed output." >&2
    exit 2  # Block the commit
  fi

  echo "✅ Tests passed! Proceeding with commit..." >&2
fi

exit 0  # Allow the action
