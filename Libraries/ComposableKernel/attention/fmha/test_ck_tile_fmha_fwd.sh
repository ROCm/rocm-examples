#!/bin/bash
# TODO: run this script from CK root or build directory
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=ComposableKernel_ck_tile_fmha_fwd
EXE=$1
KNAME=1
GPU_arch=${GPU_arch:-""}
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi

export CK_WARMUP=0
export CK_REPEAT=1

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"fmha_fwd_fails_$GPU_arch.txt"}
rm -f $CURR_FAILS_FILE
touch $CURR_FAILS_FILE
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/fmha_fwd_known_fails_$GPU_arch.txt"}

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'

# CMakeLists.txt generates only the fp16 group mode kernels for head dimensions up to 64, with LSE, without
# elementwise bias, and without the splitkv and appendkv APIs. The tests are limited to these cases:
# -prec=fp16 -mode=1, -d and -d_v up to 64, -lse=1 and -bias=n or -bias=a.

run_exe() {
    set +ex
    $EXE $@
    local ret=$?
    if [ $ret -ne 0 ] ; then
        echo "$EXE_NAME $*" >> $CURR_FAILS_FILE
    fi
    set -ex
}

run_fp16_tests() {
    local prec="fp16"
    local mode=1
    local lse=1

    for perm in 0 1 ; do
    for hdim in 32 64 ; do
    for bias in "n" "a" ; do
    for p_drop in 0.0 0.2 ; do

    run_exe -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=16    -d_v=$hdim -s=55   -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm                -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=3        -d=$hdim            -s=100  -s_k=51             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm                -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=16    -d_v=$hdim -s=99   -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=1        -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=1024 -s_k=256            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=$hdim -d_v=24    -s=3    -s_k=99             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=3 -h=2 -h_k=1 -d=$hdim            -s=200  -s_k=520            -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=t:128,30 -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=2 -h=1        -d=$hdim            -s=99   -s_k=32             -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=b:4,35   -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=33   -s_k=0              -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -kname=$KNAME $COMMON_ARGS
    run_exe -prec=$prec -mode=$mode -b=1 -h=2 -h_k=1 -d=$hdim            -s=1    -s_k=10  -s_kpad=32 -bias=$bias -p_drop=$p_drop -lse=$lse -iperm=$perm -operm=$perm -mask=2        -kname=$KNAME $COMMON_ARGS

    done ; done ; done ; done
}

run_padding_smoke_tests() {
    # Padding-only smoke tests for group mode using COMMON_ARGS
    local prec="fp16"

    # Group mode: padding via physical stride along seqlen
    local seqlens_q="1024,768,512,256"
    local seqlens_k="1024,768,512,256"
    local base_group="-prec=$prec -mode=1 -b=4 -h=16 -h_k=16 -d=64 -s=$seqlens_q -s_k=$seqlens_k -bias=n -mask=0 -lse=1 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME $COMMON_ARGS"
    # low physical pad
    run_exe $base_group -s_qpad=1152,896,576,320 -s_kpad=1152,896,576,320
    # medium physical pad
    run_exe $base_group -s_qpad=1536,1152,768,384 -s_kpad=1536,1152,768,384
    # high physical pad
    run_exe $base_group -s_qpad=2048,1536,1024,512 -s_kpad=2048,1536,1024,512
}

run_padding_basic_boundary_tests() {
    # Basic padding and boundary tests (reference: smoke_test_fwd_pad.sh)
    local prec="fp16"
    local perm

    # Group mode: Q&K padded with per-batch different strides
    for perm in 0 1 ; do
      run_exe -prec=$prec -mode=1 -b=2 -h=2 -h_k=1 -d=16 -d_v=32 \
              -s=55 -s_k=256 -s_qpad=64,60 -s_kpad=272,260 \
              -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
              -kname=$KNAME $COMMON_ARGS
    done

    # slightly larger, uneven padding strides
    for perm in 0 1 ; do
      run_exe -prec=$prec -mode=1 -b=3 -h=2 -h_k=1 -d=64 -d_v=64 \
              -s=50,60,40 -s_k=128,256,192 -s_qpad=64,64,64 -s_kpad=160,288,224 \
              -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
              -kname=$KNAME $COMMON_ARGS
    done

    # only K padded; Q unpadded
    for perm in 0 1 ; do
      run_exe -prec=$prec -mode=1 -b=2 -h=2 -h_k=1 -d=32 -d_v=64 \
              -s=55 -s_k=256 -s_kpad=272,260 \
              -bias=n -p_drop=0.0 -lse=1 -iperm=$perm -operm=$perm \
              -kname=$KNAME $COMMON_ARGS
    done

    # highly variable logical lengths
    run_exe -prec=$prec -mode=1 -b=4 -h=4 -d=32 \
            -s=1,127,3,65 -s_k=1,127,3,65 -s_kpad=128 \
            -bias=n -p_drop=0.0 -lse=1 -kname=$KNAME $COMMON_ARGS

    # GQA + Alibi + Causal mask (keep vlayout row-major for fp16/bf16
    run_exe -prec=$prec -mode=1 -b=2 -h=16 -h_k=4 -d=64 \
            -s=256,129 -s_k=256,129 -s_kpad=256 \
            -bias=a -mask=t -lse=1 -iperm=0 -operm=0 -vlayout=r \
            -kname=$KNAME $COMMON_ARGS
}

set -x

run_fp16_tests
run_padding_smoke_tests
run_padding_basic_boundary_tests

set +x

new_fails_count=0
known_fails_count=0
if [ -f $KNOWN_FAILS_FILE ] ; then
    echo "Comparing current fails ($CURR_FAILS_FILE) against known fails ($KNOWN_FAILS_FILE):"
    while IFS= read -r line; do
        if grep -Fxq "$line" $KNOWN_FAILS_FILE; then
            echo "Known fail: $line"
            known_fails_count=$(($known_fails_count + 1))
        else
            echo "New fail: $line"
            new_fails_count=$(($new_fails_count + 1))
        fi
    done < $CURR_FAILS_FILE
else
    new_fails_count=$(wc -l < $CURR_FAILS_FILE)
    echo "No known fails file, all fails ($new_fails_count) are new:"
    cat $CURR_FAILS_FILE
fi
echo "New fails count: $new_fails_count; Known fails count: $known_fails_count"
exit $(($new_fails_count != 0))
