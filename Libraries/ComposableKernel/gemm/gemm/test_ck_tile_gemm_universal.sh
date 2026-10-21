#!/bin/bash
EXE=$1
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'

# The SingleTile, Regular and LargeMatrix sizes of Composable Kernel's GEMM tests (test/ck_tile/gemm).
# The example does not pad its tiles, so the other sizes of the tests are not supported.
# The tiles are 128x128x$2, where $2 is 64 for 16-bit and 128 for 8-bit A data types.
run_tests() {
    for mnk in "128 128 $2" "512 1024 512" "2048 2048 2048"; do
        read -r m n k <<< "$mnk"

        $EXE -m=$m -n=$n -k=$k -stride_a=0 -stride_b=0 -stride_c=0 -prec=$1 $COMMON_ARGS
        if [ $? -eq 0 ]; then
            echo "Success: Test with m=$m, n=$n, k=$k executed successfully."
        else
            echo "Error: Test with m=$m, n=$n, k=$k failed to execute properly."
            exit 1
        fi

    done
}

set -x

run_tests "fp16" 64
run_tests "bf16" 64
run_tests "fp8" 128
run_tests "bf8" 128
run_tests "fp16i4" 64
run_tests "fp8i4" 128
run_tests "bf8i4" 128

set +x
