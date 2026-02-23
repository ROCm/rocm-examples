# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLES := \
  ext_op_amax \
  ext_op_layernorm \
  gemm \
  gemm_alphavec_ext \
  gemm_amax \
  gemm_amax_ext \
  gemm_amax_with_scale \
  gemm_amax_with_scale_ext \
  gemm_attr_tciA_tciB \
  gemm_batched \
  gemm_batched_ext \
  gemm_bgradb \
  gemm_bias \
  gemm_bias_ext \
  gemm_bias_swizzle_a_ext \
  gemm_clamp_bias \
  gemm_dgelu_bgradb \
  gemm_dgelu_bgradb_ext \
  gemm_ext \
  gemm_ext_bgradb \
  gemm_gelu_aux_bias \
  gemm_gelu_aux_bias_ext \
  gemm_get_algo_by_index_ext \
  gemm_get_all_algos \
  gemm_get_all_algos_ext \
  gemm_is_tuned_ext \
  gemm_mix_precision \
  gemm_mix_precision_ext \
  gemm_mix_precision_with_amax_ext \
  gemm_swish_bias \
  gemm_swizzle_a \
  gemm_swizzle_b \
  gemm_tuning_splitk_ext \
  gemm_tuning_wgm_ext \
  gemm_with_scale_a_b \
  gemm_with_scale_a_b_ext \
  gemm_with_scale_a_b_vector \
  gemm_with_TF32 \
  groupedgemm_ext \
  groupedgemm_fixed_mk_ext \
  weight_swizzle_padding


all: $(EXAMPLES)

clean: TARGET=clean
clean: all

$(EXAMPLES):
	$(MAKE) -C $@ $(TARGET)

.PHONY: all clean $(EXAMPLES)
