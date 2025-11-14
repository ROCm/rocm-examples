# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE := hip_llvm_ir_to_executable
COMMON_INCLUDE_DIR := ../../Common
GPU_RUNTIME ?= HIP

ifneq ($(GPU_RUNTIME), HIP)
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

# HIP variables
ROCM_INSTALL_DIR ?= /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX                ?= $(ROCM_INSTALL_DIR)/bin/hipcc
CLANG                 ?= $(ROCM_INSTALL_DIR)/llvm/bin/clang
LLVM_MC               ?= $(ROCM_INSTALL_DIR)/llvm/bin/llvm-mc
CLANG_OFFLOAD_BUNDLER ?= $(ROCM_INSTALL_DIR)/llvm/bin/clang-offload-bundler
LLVM_DIS              ?= $(ROCM_INSTALL_DIR)/llvm/bin/llvm-dis

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) $(CXXFLAGS)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) $(CPPFLAGS)
ILDFLAGS  := $(LDFLAGS)
ILDLIBS   := $(LDLIBS)

# Compile for these GPU architectures
HIP_ARCHITECTURES ?= gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102

# If white-space is given as a literal the `subst` cannot recognize it.
# There this `empty` `space` hack is used in the tokenizing of GPU_TARGETS
# and the creation of GPU_ARCH_TRIPLES, which is later passed to CLANG_OFFLOAD_BUNDLER
# in the targets option. The targets option needs to be a single string with no spaces.
empty =
space = $(empty) $(empty)
comma = ,

GPU_ARCHS := $(subst ;,$(space),$(HIP_ARCHITECTURES))
GPU_ARCH_TRIPLES := $(subst $(space),$(comma),$(GPU_ARCHS:%=hipv4-amdgcn-amd-amdhsa--%))

all: $(EXAMPLE)

$(EXAMPLE): main.o main_device.o
	$(HIPCXX) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

main_device.o: hip_obj_gen.mcin offload_bundle.hipfb
	$(LLVM_MC) -triple x86_64-unknown-linux-gnu -o $@ $< --filetype=obj

offload_bundle.hipfb: $(GPU_ARCHS:%=main_%.o)
	$(CLANG_OFFLOAD_BUNDLER) -type=o -bundle-align=4096 \
	  -targets=host-x86_64-unknown-linux-gnu,$(GPU_ARCH_TRIPLES) \
	  -input=/dev/null \
	  $(^:%=-input=%) \
	  -output=$@

main.o: main.hip
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -c --cuda-host-only -fuse-cuid=none $<

main_%.o: main_%.ll
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$* -o $@ $<

main_%.ll: main_%.bc
	$(LLVM_DIS) -o $@ $<

main_%.bc: main.hip
	$(HIPCXX) --cuda-device-only -c -emit-llvm --offload-arch=$* -o $@ -I ../../Common -std=c++17 $<

clean:
	rm -f \
	  main_device*.o \
	  main_*.bc \
	  offload_bundle.hipfb \
	  main_device.o \
	  main.o \
	  main_*.ll \
	  $(EXAMPLE)

.PHONY: clean $(EXAMPLE)
