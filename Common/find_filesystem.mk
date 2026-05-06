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

# Detect whether linking std::experimental::filesystem requires -lstdc++fs.
# Older toolchains (GCC <= 8 / SLES default) ship the implementation in a
# separate static library; GCC >= 9 has it merged into libstdc++ proper.
#
# Mirrors the behavior of Common/FindFilesystem.cmake for the Make build path.
#
# Usage in leaf Makefiles:
#   include $(COMMON_INCLUDE_DIR)/find_filesystem.mk
#   ...
#   ILDLIBS := -lrocdecode ... $(CXX_FS_LIB)
#
# Output variable:
#   CXX_FS_LIB - either "-lstdc++fs" or empty.

# GNU Make 4.2 treats '#' as a comment even inside $(shell), hiding the
# closing ')' from the parser.  Expand it via a variable instead.
_FS_HASH := \#

_FS_OK_BARE := $(shell { echo '$(_FS_HASH)include <experimental/filesystem>'; echo 'int main(){return std::experimental::filesystem::current_path().empty();}'; } | $${CXX:-c++} -x c++ -std=c++17 - -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(_FS_OK_BARE),0)
  _FS_OK_WITH_LIB := $(shell { echo '$(_FS_HASH)include <experimental/filesystem>'; echo 'int main(){return std::experimental::filesystem::current_path().empty();}'; } | $${CXX:-c++} -x c++ -std=c++17 - -lstdc++fs -o /dev/null 2>/dev/null && echo 1 || echo 0)
  ifeq ($(_FS_OK_WITH_LIB),1)
    CXX_FS_LIB := -lstdc++fs
  endif
endif
