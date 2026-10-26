#!/usr/bin/env bash
# Zip a solution folder with solution.py at the archive root.
#   ./make_submission.sh            -> zips my_solution/ into submission.zip
#   ./make_submission.sh baseline   -> zips baseline/
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-my_solution}"
OUT="$PWD/submission.zip"
[ -f "$SRC/solution.py" ] || { echo "no solution.py in $SRC" >&2; exit 1; }
rm -f "$OUT"
(cd "$SRC" && zip -qr "$OUT" . -x '__pycache__/*' '*.pyc' '.DS_Store')
SIZE=$(stat -f%z "$OUT")
echo "wrote submission.zip ($((SIZE / 1024)) KB) from $SRC/"
[ "$SIZE" -le $((20 * 1024 * 1024)) ] || echo "WARNING: over the 20 MB limit" >&2
unzip -l "$OUT"
