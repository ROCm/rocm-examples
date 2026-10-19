#!/bin/sh
set -e
EXE=$1

for fquant in "" "-fquant=1 -prec_o=int8" "-fquant=1 -prec_o=fp8"; do
for pr_i in "fp16" "bf16" ; do
for fadd in "0" "1"; do
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=99  -n=13
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=17  -n=16
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=1   -n=100
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=4   -n=128
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=80  -n=127
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=22  -n=255 -x_stride=256 -xr_stride=256 -y_stride=256 -yr_stride=256
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=7   -n=599
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=19  -n=512
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=33  -n=313 -x_stride=1000 -xr_stride=1000 -y_stride=1000 -yr_stride=1000
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=11  -n=510
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=171 -n=676 -x_stride=818 -xr_stride=818 -y_stride=818 -yr_stride=818
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=91  -n=636
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=12  -n=768 -x_stride=800 -xr_stride=800 -y_stride=800 -yr_stride=800
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=100 -n=766 -x_stride=812 -xr_stride=812 -y_stride=812 -yr_stride=812
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=31  -n=1024
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=64  -n=1000 -x_stride=1004 -xr_stride=1004 -y_stride=1004 -yr_stride=1004
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=8   -n=1501
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=3   -n=1826
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=5   -n=2040
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=7   -n=2734
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=1   -n=3182
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=9   -n=4096
$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=3   -n=8192
# n > 8192 uses the two-pass kernels, which generate.py builds without fused quant
if [ -z "$fquant" ]; then
$EXE -prec_i=$pr_i -fadd=$fadd -m=3   -n=9120
$EXE -prec_i=$pr_i -fadd=$fadd -m=1   -n=10547
fi
#$EXE -prec_i=$pr_i -fadd=$fadd $fquant -m=3   -n=17134
done
done
done
