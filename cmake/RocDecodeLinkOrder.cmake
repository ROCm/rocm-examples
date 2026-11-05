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

# On TheRock installs librocdecode depends on the libva vendored under
# ${ROCM_PATH}/lib/rocm_sysdeps/lib, while FFmpeg links the system libva, so both
# load at runtime. rocdecode::rocdecode is an INTERFACE target, so CMake places
# ${FFMPEG_LIBRARIES} ahead of librocdecode on the link line. The system libva
# then enters the global symbol scope first, wins VA symbol interposition, and
# vaInitialize fails (rocDecGetDecoderCaps returns ROCDEC_RUNTIME_ERROR).
#
# Call this before the target's other target_link_libraries() to make
# librocdecode the first direct dependency of the executable, as the upstream
# rocDecode samples do. It must be a raw path: CMake reorders target entries.
function(link_rocdecode_first TARGET)
    if(NOT WIN32 AND EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
        target_link_libraries(
            ${TARGET}
            PRIVATE
                -Wl,--push-state,--no-as-needed
                ${rocdecode_LIBRARY}
                -Wl,--pop-state
        )
    endif()
endfunction()
