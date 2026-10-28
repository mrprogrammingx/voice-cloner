#!/usr/bin/env bash
# Try newer package versions in a separate environment before touching .venv.
#   ./scripts/safe_update.sh          -> upgrade in .venv-test and run the tests
#   ./scripts/safe_update.sh smoke    -> also run one real (slow) generation
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON311="${PYTHON311:-/opt/homebrew/bin/python3.11}"

echo "1/4 Backing up current versions to requirements.backup.txt"
cp requirements.txt requirements.backup.txt

echo "2/4 Creating a fresh test environment (.venv-test)"
rm -rf .venv-test
"$PYTHON311" -m venv .venv-test
.venv-test/bin/python -m pip install --quiet --upgrade pip
.venv-test/bin/python -m pip install --upgrade chatterbox-tts "setuptools<81" pytest

echo "3/4 Running the tests with the new versions"
if ! .venv-test/bin/python -m pytest tests -q; then
  echo
  echo "TESTS FAILED with the new versions. Do NOT update. Your .venv is unchanged."
  exit 1
fi

if [ "${1:-}" = "smoke" ]; then
  echo "Smoke test: one real generation (takes a few minutes)"
  .venv-test/bin/python -m src.main --text "This is a short smoke test." \
    --output output/smoke.wav --no-mp3 --work-dir output/work/smoke --force
fi

echo "4/4 What would change:"
.venv-test/bin/python -m pip freeze > requirements.candidate.txt
diff requirements.txt requirements.candidate.txt || true

cat << 'MSG'

Tests passed in .venv-test. To adopt the new versions:
    source .venv/bin/activate
    pip install -r requirements.candidate.txt
    cp requirements.candidate.txt requirements.txt
To go back at any time:
    pip install -r requirements.backup.txt
Then generate one short script and listen before relying on it.
MSG
