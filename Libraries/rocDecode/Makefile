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

EXAMPLES := \
	rocdec_decode \
	video_decode \
	video_decode_batch \
	video_decode_mem \
	video_decode_multi_files \
	video_decode_perf \
	video_decode_pic_files \
	video_decode_raw \
	video_decode_rgb \
	video_to_sequence

# All rocDecode tests are skipped by default: they require test data
# (videos under $(ROCM_PATH)/share/rocdecode/) that ctest guards behind
# `if(EXISTS …)` checks. Encoding the same conditional logic in Make is
# possible but noisy; for now skip the whole subtree so `make test` from
# root stays clean. To run rocDecode tests manually, cd into the leaf
# Makefile and run `make test`.
SKIP_FROM_TEST := $(EXAMPLES)

all: $(EXAMPLES)

clean: TARGET=clean
clean: all

$(EXAMPLES):
	$(MAKE) -C $@ $(TARGET)

test: TARGET=test
test: $(filter-out $(SKIP_FROM_TEST),$(EXAMPLES))

.PHONY: all clean test $(EXAMPLES)
