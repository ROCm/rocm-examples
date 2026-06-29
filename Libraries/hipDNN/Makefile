# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
	bn_inference \
	bn_inference_with_variance \
	bn_training \
	bn_backward \
	fused_bn_training_activ \
	fused_bn_inference_activ \
	fused_bn_inference_variance_activ \
	fused_bn_inf_drelu_bn_bwd \
	conv_fprop \
	conv_dgrad \
	conv_wgrad \
	conv_fprop_deterministic \
	fused_conv_fprop_activ \
	fused_conv_fprop_bias_activ \
	matmul

SKIP_FROM_TEST :=

all: $(EXAMPLES)

clean: TARGET=clean
clean: all

$(EXAMPLES):
	$(MAKE) -C $@ $(TARGET)

test: TARGET=test
test: $(filter-out $(SKIP_FROM_TEST),$(EXAMPLES))

.PHONY: all clean test $(EXAMPLES)
