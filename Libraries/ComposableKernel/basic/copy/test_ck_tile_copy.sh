#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

BIN="$1"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-100}"
VALIDATE="${VALIDATE:-1}"

MS=(128 256 512 1024)
NS=(64 256 1024 2048 4096)
PRECS=(fp16 fp32)

echo "Using BIN=$BIN"
echo "WARMUP=$WARMUP REPEAT=$REPEAT VALIDATE=$VALIDATE"

failures=0

for prec in "${PRECS[@]}"; do
  for m in "${MS[@]}"; do
    for n in "${NS[@]}"; do
      echo "=============================================="
      echo "Running: prec=$prec m=$m n=$n"
      set +e
      out="$("$BIN" -prec="$prec" -m="$m" -n="$n" -warmup="$WARMUP" -repeat="$REPEAT" -v="$VALIDATE" 2>&1)"
      rc=$?
      set -e

      echo "$out"
      if [[ $rc -ne 0 ]]; then
        echo "RUN ERROR for m=$m n=$n prec=$prec"
        ((failures++)) || true
        continue
      fi

      if [[ "$VALIDATE" == "1" ]]; then
        if ! grep -q "valid:y" <<<"$out"; then
          echo "VALIDATION FAILED for m=$m n=$n prec=$prec"
          ((failures++)) || true
        fi
      fi
    done
  done
done

echo "=============================================="
if [[ $failures -eq 0 ]]; then
  echo "All runs passed"
else
  echo "$failures runs failed"
  exit 1
fi