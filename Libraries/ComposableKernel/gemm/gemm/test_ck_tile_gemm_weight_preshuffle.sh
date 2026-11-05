#!/bin/bash
EXE=$1
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=2 -warmup=0 -repeat=1'

# The sizes of Composable Kernel's weight preshuffle GEMM tests (test/ck_tile/gemm_weight_preshuffle)
run_tests() {
    for mnk in "2048 4096 5120" \
        "128 128 128" "128 128 4096" "128 2048 128" "128 2048 4096" \
        "1024 128 128" "1024 128 4096" "1024 2048 128" "1024 2048 4096"; do
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

run_tests "fp16"
run_tests "bf16"
run_tests "fp8"
run_tests "bf8"

set +x
