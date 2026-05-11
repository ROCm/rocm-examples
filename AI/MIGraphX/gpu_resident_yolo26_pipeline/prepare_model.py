# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Export YOLO26s to ONNX and compile it for MIGraphX.

The export step must run in a separate interpreter from the compile step:
Ultralytics' ONNX export sets ``CUDA_VISIBLE_DEVICES=''`` to force a CPU
trace, and HIP caches the device list at first use, so a subsequent
``migraphx.get_target("gpu")`` in the same process raises
``get_device_id: No device``. The subprocess fence below isolates that
side effect; the compile step then runs against a clean HIP environment.
"""

import subprocess
import sys

# --- Step 0a: Export YOLO26s to ONNX
subprocess.check_call([
    sys.executable, "-c",
    "from ultralytics import YOLO; "
    "YOLO('yolo26s.pt').export(format='onnx', dynamic=False, batch=1, imgsz=640)",
])

# --- Step 0b: Compile ONNX -> MIGraphX .mxr (article snippet, verbatim)
import migraphx

model = migraphx.parse_onnx("yolo26s.onnx")

# FP16 quantization speeds up model inference without accuracy drop
migraphx.quantize_fp16(model)

# offload_copy=False exposes the output as a named parameter so we can bind
# a pre-allocated PyTorch tensor to it at inference time (see Step 3).
model.compile(migraphx.get_target("gpu"), offload_copy=False)

migraphx.save(model, "model.mxr")
