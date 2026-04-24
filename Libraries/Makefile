# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

LIBRARIES := \
	hipBLAS \
	hipCUB \
	hipFFT \
	hipSOLVER \
	hipSPARSE \
	hipSPARSELt \
	rocRAND \
	hipRAND

SKIP_FROM_TEST :=

ifneq ($(GPU_RUNTIME), CUDA)

ROCM_PATH ?= /opt/rocm
INCLUDE_DIR := $(ROCM_PATH)/include

LIBRARIES += \
  hipBLASLt \
  RCCL \
  rocBLAS \
  rocFFT \
  rocPRIM \
  rocProfiler-SDK \
  rocSOLVER \
  rocSPARSE \
  rocThrust \
  rocWMMA

ifneq ($(wildcard $(INCLUDE_DIR)/rocdecode/rocdecode.h),)
LIBRARIES += rocDecode
endif

ifneq ($(wildcard $(INCLUDE_DIR)/rocjpeg/rocjpeg.h),)
LIBRARIES += rocJPEG
endif

# Only build libraries whose headers are present
ifneq ($(wildcard $(INCLUDE_DIR)/hiptensor/hiptensor.hpp),)
LIBRARIES += hipTensor
endif

ifneq ($(wildcard $(INCLUDE_DIR)/migraphx/migraphx.hpp),)
LIBRARIES += MIGraphX
endif

ifneq ($(wildcard $(INCLUDE_DIR)/mivisionx/VX/vx.h),)
LIBRARIES += MIVisionX
endif

ifneq ($(wildcard $(INCLUDE_DIR)/rocal/rocal_api.h),)
LIBRARIES += rocAL
endif

ifneq ($(wildcard $(INCLUDE_DIR)/rocalution/rocalution.hpp),)
LIBRARIES += rocALUTION
endif

ifneq ($(wildcard $(INCLUDE_DIR)/roccv/core/tensor.hpp),)
LIBRARIES += rocCV
endif

ifneq ($(wildcard $(INCLUDE_DIR)/rpp/rpp.h),)
LIBRARIES += RPP
endif

endif

all: $(LIBRARIES)

clean: TARGET=clean
clean: all

$(LIBRARIES):
	$(MAKE) -C $@ $(TARGET)

test: TARGET=test
test: $(filter-out $(SKIP_FROM_TEST),$(LIBRARIES))

.PHONY: all clean test $(LIBRARIES)
