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
    add_kernel \
    asynchronous_kernel_execution \
    call_stack_management \
    complex_math \
    constant_memory \
    data_prefetching \
    device_enumeration \
    device_recursion \
    device_selection \
    dynamic_shared_memory \
    dynamic_unified_memory \
    error_handling \
    event_based_asynchronous_kernel_execution \
    explicit_copy \
    explicit_memory \
    kernel_memory_allocation \
    low_precision_float_fp8 \
    low_precision_float_fp16 \
    math \
    memory_range_attributes \
    memory_pool \
    memory_pool_interprocess_handle \
    memory_pool_interprocess_pointer \
    memory_pool_resource_usage_statistics \
    memory_pool_treshold \
    memory_pool_trim \
    multi_device_synchronization \
    ordinary_memory_allocation \
    p2p_memory_access \
    p2p_memory_access_failed \
    pageable_host_memory \
    pinned_host_memory \
    sequential_kernel_execution \
    simple_device_query \
    standard_unified_memory \
    static_shared_memory \
    static_unified_memory \
    stream_ordered_memory_allocation \
    unified_memory_advice \
    virtual_memory

ifeq ($(GPU_RUNTIME), CUDA) # Only supported on CUDA
    EXAMPLES += \
        block_reduction
endif

all: $(EXAMPLES)

clean: TARGET=clean
clean: all

$(EXAMPLES):
    $(MAKE) -C $@ $(TARGET)

.PHONY: all clean $(EXAMPLES)
