#!/usr/bin/env bash
#
# A2A Composability Demo — a consumer agent hires the Token Trust Analyzer over CAP.
#
# One command, no keys, no funded wallet: in simulation mode the consumer narrates
# the full Post -> Lock -> Deliver -> Clear loop and prints the delivered Trust
# Report. If a local analyzer API is running it hires it over HTTP; otherwise it
# runs the pipeline in-process so the demo always completes.
#
#   ./scripts/demo_a2a.sh                 # DAI, auto mode (simulation unless CROO is set)
#   ./scripts/demo_a2a.sh 0x<addr>        # a different token
#   MODE=simulation ./scripts/demo_a2a.sh # force simulation
#   MODE=live ./scripts/demo_a2a.sh       # force live CROO (needs the CROO_* env)
#
# For the "hire over HTTP" flavor, start the analyzer first in another shell:
#   python app.py
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the project venv if present, else fall back to `python`.
if [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

ADDR="${1:-0x6B175474E89094C44Da98b954EedeAC495271d0F}"  # DAI
MODE="${MODE:-auto}"

echo "=================================================================="
echo " CROO A2A demo — consumer agent hires the Token Trust Analyzer"
echo " token: ${ADDR}   mode: ${MODE}"
echo "=================================================================="
exec "$PYTHON" -m cap.consumer "$ADDR" --chain "${CHAIN:-ethereum}" --mode "$MODE"
