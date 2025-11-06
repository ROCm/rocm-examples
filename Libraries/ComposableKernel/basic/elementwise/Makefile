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

EXAMPLE := ComposableKernel_ck_tile_elementwise
EXAMPLE_UNARY := ComposableKernel_ck_tile_elementwise_unary
EXAMPLE_TRANSPOSE := ComposableKernel_ck_tile_elementwise_transpose
EXAMPLE_ADD_4D := ComposableKernel_ck_tile_elementwise_add_4d
GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) -x hip
ICPPFLAGS :=
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): elementwise_example.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

$(EXAMPLE_UNARY): elementwise_example_unary.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

$(EXAMPLE_TRANSPOSE): elementwise_example_transpose.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

$(EXAMPLE_ADD_4D): elementwise_example_add_4d.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE) $(OBJECTS)

.PHONY: clean
