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

EXAMPLE := mivisionx_mv_objdetect
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External

# ROCm installation directory
ROCM_INSTALL_DIR := /opt/rocm

# Batch size configuration (can be overridden: make BATCH_SIZE=4)
BATCH_SIZE := 1

# Build directory for generated files
BUILD_DIR := tmp

# Model configuration
MODEL_URL := https://github.com/kiritigowda/YoloV2NCS/raw/master/models/caffemodels/yoloV2Tiny20.caffemodel
MODEL_FILE := $(BUILD_DIR)/yoloV2Tiny20.caffemodel

# MIVisionX and OpenCV include directories
MIVISIONX_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include/mivisionx
OPENCV_INCLUDE_DIR := /usr/include/opencv4

CXX ?= g++

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) -msse4.2 -mf16c
ICPPFLAGS := -I $(ROCM_INSTALL_DIR)/include -I $(MIVISIONX_INCLUDE_DIR) -I $(OPENCV_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR)
ICPPFLAGS += -I . -I $(BUILD_DIR)/mv_extras -I $(BUILD_DIR)
ICPPFLAGS += -DENABLE_MVDEPLOY=1 -DENABLE_OPENCV=1
ICPPFLAGS += -DEXAMPLE_DATA_DIR=\"$(CURDIR)/data\" -DMV_INSTALL_FOLDER=\"$(CURDIR)/$(BUILD_DIR)\"
ILDFLAGS  := -L $(ROCM_INSTALL_DIR)/lib -L $(BUILD_DIR)/lib
ILDLIBS   := -lopenvx -lvx_nn -lvx_amd_media -lmv_deploy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lpthread -ldl

CXXFLAGS ?= -Wall -Wextra

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

# Check if mv_compile exists
MV_COMPILE := $(ROCM_INSTALL_DIR)/bin/mv_compile
ifeq (,$(wildcard $(MV_COMPILE)))
    $(error mv_compile not found at $(MV_COMPILE). Please install MIVisionX with model compiler support)
endif

# Generated files (in build directory)
GENERATED_FILES := $(BUILD_DIR)/lib/libmv_deploy.so \
                   $(BUILD_DIR)/mvdeploy_api.cpp \
                   $(BUILD_DIR)/mvdeploy_api.h \
                   $(BUILD_DIR)/mvdeploy.h \
                   $(BUILD_DIR)/weights.bin \
                   $(BUILD_DIR)/mv_extras/mv_extras_postproc.cpp

.PHONY: all clean

all: $(EXAMPLE)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Download model if not present
$(MODEL_FILE): | $(BUILD_DIR)
	@echo "Downloading YoloV2 model..."
	@wget -q --show-progress $(MODEL_URL) -O $(MODEL_FILE) || (echo "Failed to download model" && exit 1)
	@echo "Model downloaded successfully"

# Run mv_compile to generate deployment artifacts
$(GENERATED_FILES): $(MODEL_FILE)
	@echo "Compiling YoloV2 model with mv_compile (batch size: $(BATCH_SIZE))..."
	@cd $(BUILD_DIR) && CC=gcc CXX=g++ $(MV_COMPILE) --model yoloV2Tiny20.caffemodel --install_folder . --input_dims $(BATCH_SIZE),3,416,416
	@echo "Model compilation complete"

# Build the example
$(EXAMPLE): main.cpp visualize.cpp $(BUILD_DIR)/mvdeploy_api.cpp $(BUILD_DIR)/mv_extras/mv_extras_postproc.cpp $(COMMON_INCLUDE_DIR)/mivisionx_utils.hpp $(GENERATED_FILES)
	$(CXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ main.cpp visualize.cpp $(BUILD_DIR)/mvdeploy_api.cpp $(BUILD_DIR)/mv_extras/mv_extras_postproc.cpp $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)
	$(RM) -r $(BUILD_DIR)

# The Python scripts used by MIVisionX don't tolerate parallel Makefile builds.
# The file generator must be run serially.
.NOTPARALLEL: $(GENERATED_FILES)
