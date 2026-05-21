# rocDecode and MIGraphX Zero-Copy YOLO Video Inference (Python)

![YOLO26 detections (person, bicycle) drawn on a frame from a peloton of cyclists](images/result_boxes.jpg)

## Description

This example keeps a video frame on the GPU from decode through to detection
on AMD hardware. The on-chip VCN engine decodes the bitstream via
[rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/),
[DLPack](https://github.com/dmlc/dlpack) hands the decoded surface to PyTorch
as a zero-copy view, YOLO preprocessing runs on the active HIP stream, and
[MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/) runs
the compiled YOLO26 graph on that same stream. Only the surviving detections
cross the PCIe bus. An OpenCV CPU-decode path is provided as a baseline.

## Application Flow

1. Load the compiled MIGraphX `.mxr` model and open the input video.
2. For every frame:
   1. **Decode.** rocDecode demuxes the bitstream and the VCN engine decodes
      to a GPU surface. The `opencv` baseline does this on the CPU.
   2. **DLPack wrap.** The decoded RGB surface is exposed to PyTorch as a
      zero-copy `[H, W, 3]` uint8 CUDA tensor.
   3. **Preprocess.** HWC uint8 RGB is permuted to BCHW float32, then
      resized and letterbox-padded to 640x640 on the active HIP stream.
   4. **Inference.** `migraphx.run_async` is enqueued on the same stream as
      preprocessing, so no implicit synchronization happens between them.
   5. **Postprocess.** Confidence filter and letterbox-to-source remap stay
      on-device. Only the surviving boxes are copied back to the host.
3. Draw boxes on each frame and write an MP4 with detections.

## Key APIs and Concepts

- **Zero-copy decoder-to-PyTorch handoff.** rocPyDecode exposes its decoded
  surface through DLPack, and `torch.from_dlpack` wraps it as a CUDA tensor
  that shares the same memory. There is no `hipMemcpy` between decode and
  preprocess.
- **Single HIP stream for preprocess and inference.** `Detector` queries
  `torch.cuda.current_stream()` once and reuses it for `run_async`, so
  preprocess and inference serialize on the GPU without host synchronization.
- **Pre-allocated MIGraphX output.** The output tensor is allocated once
  with `torch.empty_strided` and bound to MIGraphX as a raw pointer via
  `migraphx.argument_from_pointer`, avoiding per-frame device allocation.

## Demonstrated API Calls

### rocDecode (Python bindings)

- `pyRocVideoDecode.demuxer.demuxer`
- `pyRocVideoDecode.decoder.decoder`
- `decoder.DecodeFrame`, `decoder.GetFrameRgb`, `decoder.ReleaseFrame`
- `demuxer.DemuxFrame`, `demuxer.GetCodecId`, `demuxer.GetBitDepth`

### MIGraphX (Python)

- `migraphx.parse_onnx`, `migraphx.quantize_fp16`, `migraphx.save`,
  `migraphx.load`
- `migraphx.program.run_async`, `migraphx.argument_from_pointer`
- `migraphx.get_target`

### PyTorch and DLPack

- `torch.from_dlpack`
- `torch.cuda.current_stream`, `torch.cuda.Stream.synchronize`
- `torch.empty_strided`, `torch.nn.functional.interpolate`,
  `torch.nn.functional.pad`

## Setup

Run inside the official ROCm PyTorch container. PyTorch and MIGraphX are
pre-installed; rocDecode and its libva backend are installed below.

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size=8g \
    -v "$PWD":/workspace -w /workspace \
    rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.10.0
```

Inside the container:

```bash
# The AMDGPU "graphics" repo provides libva-amdgpu / mesa-amdgpu VA drivers
# that rocDecode needs at runtime but the rocm/pytorch image does not ship.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/graphics/7.2.3/ubuntu jammy main" \
    > /etc/apt/sources.list.d/amdgpu-graphics.list

apt-get update
apt-get install -y --no-install-recommends \
    rocdecode rocpydecode rocdecode-host \
    libva-amdgpu-drm2 mesa-amdgpu-va-drivers

# rocDecode Python bindings ship in /opt/rocm/lib.
export PYTHONPATH=/opt/rocm/lib

pip install -r requirements.txt
python3 prepare_model.py    # exports YOLO26s ONNX, compiles to model.mxr
```

## Run

```bash
python3 main.py --decoder rocdecode --input data/peloton_sample_ai_gen.mp4 --output output.mp4
python3 main.py --decoder opencv    --input data/peloton_sample_ai_gen.mp4 --output output_cv.mp4
```

`main.py --help` lists every flag. The script prints average per-frame
`predict()` and full-pipeline latencies on exit.
