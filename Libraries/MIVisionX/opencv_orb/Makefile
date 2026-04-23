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

EXAMPLE := mivisionx_opencv_orb
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External

# ROCm installation directory
ROCM_PATH ?= /opt/rocm

# MIVisionX and OpenCV include directories
MIVISIONX_INCLUDE_DIR := $(ROCM_PATH)/include/mivisionx
OPENCV_INCLUDE_DIR := /usr/include/opencv4

CXX ?= g++

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(MIVISIONX_INCLUDE_DIR) -I $(OPENCV_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR)
ICPPFLAGS += -DEXAMPLE_DATA_DIR=\"$(CURDIR)/../images\"
ILDFLAGS  := -L $(ROCM_PATH)/lib
ILDLIBS   := -lopenvx -lvx_opencv -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

CXXFLAGS ?= -Wall -Wextra

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp $(COMMON_INCLUDE_DIR)/mivisionx_utils.hpp
	$(CXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean
