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

# Architecture filtering for HIP Makefiles
#
# Usage in example Makefiles:
#   SUPPORTED_HIP_ARCHITECTURES := gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201
#   include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk
#
# After including this file, use:
#   - OFFLOAD_ARCH_FLAGS: contains --offload-arch flags for hipcc
#   - SKIP_BUILD: set to 1 if no supported architectures remain
#
# This filtering only applies if HIP_ARCHITECTURES is explicitly set by the user.
# If HIP_ARCHITECTURES is not set, hipcc will use its default behavior.

# Only perform filtering if HIP_ARCHITECTURES is explicitly set
ifdef HIP_ARCHITECTURES

# Convert semicolon-separated list to space-separated
_REQUESTED_ARCHS := $(subst ;, ,$(HIP_ARCHITECTURES))

# Filter: keep only architectures that are in both lists
FILTERED_HIP_ARCHITECTURES := $(filter $(SUPPORTED_HIP_ARCHITECTURES),$(_REQUESTED_ARCHS))

# Compute removed architectures
REMOVED_HIP_ARCHITECTURES := $(filter-out $(SUPPORTED_HIP_ARCHITECTURES),$(_REQUESTED_ARCHS))

# Print warning if architectures were removed
ifneq ($(REMOVED_HIP_ARCHITECTURES),)
  $(warning $(EXAMPLE): Removing unsupported architectures: $(REMOVED_HIP_ARCHITECTURES))
endif

# Set skip flag if no supported architectures remain
ifeq ($(FILTERED_HIP_ARCHITECTURES),)
  SKIP_BUILD := 1
  $(info $(EXAMPLE): No supported architectures found. Skipping build.)
else
  # Build --offload-arch flags for hipcc
  OFFLOAD_ARCH_FLAGS := $(foreach arch,$(FILTERED_HIP_ARCHITECTURES),--offload-arch=$(arch))
endif

else
  # HIP_ARCHITECTURES not set - use hipcc defaults, no filtering needed
  OFFLOAD_ARCH_FLAGS :=
  SKIP_BUILD := 0
endif
