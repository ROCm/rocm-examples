# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""GPU-resident YOLO video inference on AMD GPUs.

From decode to detection every frame stays in VRAM; only the surviving
boxes (and, for the demo, the raw RGB frame used to draw the output MP4)
cross the PCIe bus. Two decoder paths are provided:

* ``rocdecode`` pulls a zero-copy DLPack view out of rocPyDecode and feeds
  it into MIGraphX on the active HIP stream.
* ``opencv`` decodes on the CPU and serves as a same-machine baseline.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import migraphx
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track


# ============================================================================
# Constants
# ============================================================================

TARGET_INPUT_SIZE = 640                # YOLO26 expects square 640x640 input
LETTERBOX_FILL_NORM = 114.0 / 255.0    # YOLO letterbox padding value (RGB, normalised)
DEFAULT_DEVICE_ID = 0                  # GPU index passed to rocDecode
MP4_FOURCC = "mp4v"                    # cv2.VideoWriter codec
ROC_RGB_FORMAT_RGB = 3                 # pyRocVideoDecode rgb_format enum: RGB packed

# Approximate CCIR 601 luma weights (B, G, R) used to pick black/white text on a coloured background.
LUMA_WEIGHTS_BGR = (0.114, 0.587, 0.299)
LUMA_TEXT_DARK_THRESHOLD = 140

# Deterministic per-class colour palette (BGR); same class keeps the same tint across frames.
PALETTE_BGR = [
    (255, 119, 46),  (180, 119, 31),  (14, 127, 255),  (44, 160, 44),
    (40, 39, 214),   (189, 103, 148), (75, 86, 140),   (127, 127, 127),
    (34, 189, 188),  (207, 190, 23),  (232, 176, 174), (120, 187, 255),
    (150, 218, 152), (148, 156, 255), (156, 158, 199), (207, 199, 196),
    (219, 219, 197), (229, 218, 158), (165, 214, 197), (154, 204, 219),
]


@dataclass(frozen=True)
class Detection:
    """One YOLO detection: class id/name, confidence, and ``(x, y, w, h)`` box in pixels."""

    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VideoInfo:
    """Probed video metadata: fps, frame size, and total frame count."""

    fps: int
    width: int
    height: int
    total_frames: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames"


@dataclass
class Timings:
    """Per-frame timing accumulator for one run.

    Two-bucket design, symmetric across decoder backends:

    * ``predict_s`` covers Step 2-4 (preprocess + inference +
      postprocess) inside ``Detector.detect_on_gpu``. Identical scope
      in both branches.
    * ``pipeline_s`` is the wall-clock time of decode + ``predict``:
      ``DemuxFrame + DecodeFrame + GetFrameRgb + DLPack-wrap +
      detect_on_gpu`` for rocdecode, ``cap.read() + cvtColor +
      detect_on_gpu`` for opencv.

    Demo-only host-side work (raw-frame DtoH, ``cv2.cvtColor`` for the
    writer, ``draw_detections``, ``writer.write``) is intentionally
    excluded from both buckets: in a detection-only deployment those
    steps disappear, and including them would conflate the
    GPU-resident pipeline with the visualization harness.
    """

    frames: int = 0
    predict_s: float = 0.0     # time inside Detector.detect_on_gpu()
    pipeline_s: float = 0.0    # wall-clock per frame (decode + predict + draw)

    def report(self, pipeline_label: str, output_path: str | Path) -> None:
        """Print average ms/frame and fps for predict() and the full pipeline."""
        if self.frames == 0:
            return
        predict_ms = self.predict_s / self.frames * 1000
        pipeline_ms = self.pipeline_s / self.frames * 1000
        print(f"\n{'=' * 60}")
        print(f"Processing complete!  Total frames: {self.frames}")
        print(f"Average predict():         {predict_ms:.2f} ms  ({self.frames / self.predict_s:.1f} fps)")
        print(f"Average {pipeline_label}: {pipeline_ms:.2f} ms  ({self.frames / self.pipeline_s:.1f} fps)")
        print(f"Output saved to: {output_path}")
        print(f"{'=' * 60}\n")


# ============================================================================
# Detector: model state + Step 2/3/4 helpers
# ============================================================================

class Detector:
    """GPU-resident YOLO26 detector backed by a compiled MIGraphX .mxr model.

    Encapsulates the model state and the Step 2 (preprocess), Step 3 (inference),
    and Step 4 (postprocess) operations of the pipeline.
    """

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25):
        """Load the compiled MIGraphX model and pre-allocate its GPU output buffer."""
        self.model = migraphx.load(str(model_path))
        self.classes = self.COCO_CLASSES
        self.conf_threshold = conf_threshold
        param_shapes = self.model.get_parameter_shapes()
        self.input_name = "images"
        self.output_name = next(name for name in param_shapes if name != self.input_name)
        self.input_shape = param_shapes[self.input_name]
        self.output_shape = param_shapes[self.output_name]
        self.output_tensor = torch.empty_strided(
            self.output_shape.lens(), self.output_shape.strides(), dtype=torch.float32, device="cuda"
        )
        self.mgx_output_arg = migraphx.argument_from_pointer(
            self.output_shape, self.output_tensor.data_ptr()
        )

    # --- Step 2: Preprocessing ------------------------------------------------
    def preprocess_color_layout(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Convert HWC uint8 RGB to BCHW float32 in [0, 1]."""
        return rgb_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    @staticmethod
    def letterbox_geometry(h: int, w: int, target: int = TARGET_INPUT_SIZE) -> tuple[float, int, int]:
        """Return uniform scale and symmetric letterbox padding for *target*x*target*."""
        scale = min(target / w, target / h)
        pad_x = (target - int(w * scale)) // 2
        pad_y = (target - int(h * scale)) // 2
        return scale, pad_x, pad_y

    def preprocess_spatial(self, tensor: torch.Tensor, target: int = TARGET_INPUT_SIZE) -> torch.Tensor:
        """Resize to scaled size and letterbox-pad to *target*x*target*."""
        h, w = tensor.shape[2], tensor.shape[3]
        scale, pad_x, pad_y = self.letterbox_geometry(h, w, target)
        new_h, new_w = int(h * scale), int(w * scale)
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        padding = (pad_x, target - new_w - pad_x, pad_y, target - new_h - pad_y)
        return F.pad(tensor, padding, value=LETTERBOX_FILL_NORM).contiguous()

    # --- Step 3: Inference ----------------------------------------------------
    def run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Enqueue MIGraphX inference on the active PyTorch stream."""
        curr_stream = torch.cuda.current_stream()
        mgx_buffers = {
            self.input_name: migraphx.argument_from_pointer(self.input_shape, input_tensor.data_ptr()),
            self.output_name: self.mgx_output_arg,
        }
        self.model.run_async(mgx_buffers, curr_stream.cuda_stream, "ihipStream_t")
        return self.output_tensor

    # --- Step 4: Postprocessing ----------------------------------------------
    @staticmethod
    def filter_predictions(raw: torch.Tensor, conf_thresh: float) -> torch.Tensor:
        """Keep predictions above *conf_thresh* (returns [N, 6])."""
        preds = raw[0]
        mask = preds[:, 4] > conf_thresh
        return preds[mask].clone()

    @staticmethod
    def transform_coordinates(
        survivors: torch.Tensor, scale: float, pad_x: int, pad_y: int
    ) -> torch.Tensor:
        """Map boxes from letterboxed model space back to the source frame."""
        survivors[:, [0, 2]] = (survivors[:, [0, 2]] - pad_x) / scale
        survivors[:, [1, 3]] = (survivors[:, [1, 3]] - pad_y) / scale
        return survivors

    # --- Pipeline -------------------------------------------------------------
    def detect_on_gpu(self, rgb_tensor: torch.Tensor | np.ndarray) -> list[Detection]:
        """Run Step 2 to 4 on a GPU-resident RGB frame and return detections.

        *rgb_tensor* may be either a [H, W, 3] uint8 CUDA tensor (rocDecode path)
        or a host numpy array (OpenCV baseline); the latter is uploaded to GPU once.
        """
        if not isinstance(rgb_tensor, torch.Tensor):
            rgb_tensor = torch.from_numpy(rgb_tensor).cuda()

        # Step 2: Preprocessing (all on GPU, no host copies)
        assert rgb_tensor.is_cuda, "decoder returned a CPU tensor"
        chw = self.preprocess_color_layout(rgb_tensor)
        scale, pad_x, pad_y = self.letterbox_geometry(chw.shape[2], chw.shape[3])
        model_input = self.preprocess_spatial(chw)
        assert model_input.is_cuda, "preprocess produced a CPU tensor"

        # Step 3: Inference -> [1, 300, 6] = [x1, y1, x2, y2, conf, class_id]
        raw = self.run_inference(model_input)
        assert raw.is_cuda, "MIGraphX output is not on the GPU"

        # Step 4: Postprocessing -> survivors moved off-device in one batched DtoH.
        # YOLO26's end-to-end head already emits up to 300 final detections
        # (no NMS needed); we only filter by confidence and undo letterbox.
        # For YOLO11 and earlier, replace filter_predictions() with
        # torchvision.ops.batched_nms or ultralytics.utils.ops.non_max_suppression.
        survivors = self.filter_predictions(raw, conf_thresh=self.conf_threshold)
        survivors = self.transform_coordinates(survivors, scale, pad_x, pad_y)

        # MIGraphX writes via run_async on the active PyTorch stream; barrier
        # ensures the output buffer is ready before the DtoH copy below.
        torch.cuda.current_stream().synchronize()
        host = survivors.cpu().numpy()

        detections: list[Detection] = []
        for x1, y1, x2, y2, conf, cid in host:
            cid = int(cid)
            class_name = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
            detections.append(Detection(
                class_id=cid,
                class_name=class_name,
                confidence=float(conf),
                box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            ))
        return detections


# ============================================================================
# Drawing, DLPack shim, video writer, stats
# ============================================================================

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """Draw bounding boxes and confidence labels onto *frame* in-place (BGR)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    wb, wg, wr = LUMA_WEIGHTS_BGR
    for det in detections:
        x, y, w, h = det.box
        colour = PALETTE_BGR[det.class_id % len(PALETTE_BGR)]
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        (lw, lh), bl = cv2.getTextSize(label, font, 0.5, 1)
        b, g, r = colour
        text_colour = (20, 20, 20) if wb * b + wg * g + wr * r > LUMA_TEXT_DARK_THRESHOLD else (255, 255, 255)
        cv2.rectangle(frame, (x, y - lh - bl - 6), (x + lw + 4, y), colour, -1)
        cv2.putText(frame, label, (x + 2, y - bl - 2), font, 0.5, text_colour, 1, cv2.LINE_AA)


def decoded_rgb_view(packet) -> torch.Tensor:
    """Wrap rocPyDecode's RGB surface as a correctly-strided [H-1, W, 3] PyTorch view.

    Workaround for a rocPyDecode 0.8.0 bug: the DLPack capsule advertises strides
    ``(W*3, 1, 0)`` for an ``[H, W, 3]`` shape (collapsing R/G/B onto one byte).
    The buffer itself is packed HWC RGB but is one row short of ``H*W*3``; we
    rebuild the strides as ``(W*3, 3, 1)`` and clamp height to ``H-1``.
    """
    raw = torch.from_dlpack(packet.ext_buf[0])
    H, W = raw.shape[:2]
    return raw.as_strided((H - 1, W, 3), (W * 3, 3, 1))


def _open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open *path* with OpenCV and return the capture together with its metadata.

    Raises ``ValueError`` if the video cannot be opened. The caller owns the
    returned capture and is responsible for releasing it.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, info


def _make_writer(path: str | Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    """Create an MP4 writer at *path* for ``(width, height)`` frames at *fps*."""
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*MP4_FOURCC), fps, (width, height))


# ============================================================================
# Run modes
# ============================================================================

def process_video_rocdecode(
    detector: Detector, input_path: str | Path, output_path: str | Path
) -> None:
    """Run the full GPU pipeline: VCN decode -> DLPack -> Detector -> draw -> encode."""
    try:
        import pyRocVideoDecode.decoder as rocdec
        import pyRocVideoDecode.demuxer as rocdmx
        import pyRocVideoDecode.types as rocdectypes
    except Exception as e:
        raise RuntimeError(
            "pyRocVideoDecode is not available. Ensure rocdecode/rocPyDecode are installed and set PYTHONPATH=/opt/rocm/lib"
        ) from e

    # cv2 is used only for metadata; the writer is sized from the first decoded
    # frame to account for the rocPyDecode stride bug (may be 1 row shorter).
    cap, info = _open_video(input_path)
    cap.release()
    print(f"\nVideo info: {info}")

    writer: cv2.VideoWriter | None = None  # created lazily on first decoded frame

    demux = rocdmx.demuxer(str(input_path))
    codec_id = rocdec.GetRocDecCodecID(demux.GetCodecId())

    viddec = rocdec.decoder(
        codec_id,
        device_id=DEFAULT_DEVICE_ID,
        mem_type=rocdectypes.OUT_SURFACE_MEM_DEV_COPIED,
        b_force_zero_latency=False,
        crop_rect=None,
        max_width=0,
        max_height=0,
        clk_rate=1000,
    )

    if not viddec.IsCodecSupported(DEFAULT_DEVICE_ID, codec_id, demux.GetBitDepth()):
        cfg = viddec.GetGpuInfo()
        raise RuntimeError("Codec is not supported on this GPU " + cfg.device_name)

    print("Decoding started, please wait...")

    timings = Timings()

    while True:
        # Step 1: Decoding: VCN demuxes and decodes the compressed bitstream on-chip
        demux_decode_start = time.perf_counter()
        packet = demux.DemuxFrame()
        n_frame_returned = viddec.DecodeFrame(packet)
        timings.pipeline_s += time.perf_counter() - demux_decode_start

        for _ in range(n_frame_returned):  # one packet may yield 0-N frames
            frame_process_start = time.perf_counter()
            pts = viddec.GetFrameRgb(packet, rgb_format=ROC_RGB_FORMAT_RGB)  # NV12->RGB on the GPU via HIP kernel
            if pts == -1:
                viddec.ReleaseFrame(packet)
                continue

            rgb_tensor = decoded_rgb_view(packet)  # [H, W, 3] uint8, cuda

            predict_start = time.perf_counter()
            detections = detector.detect_on_gpu(rgb_tensor)
            timings.predict_s += time.perf_counter() - predict_start
            timings.pipeline_s += time.perf_counter() - frame_process_start
            timings.frames += 1

            # Demo-only host-side work, excluded from pipeline_s:
            # the GPU-resident pipeline ends at detect_on_gpu (only
            # survivors crossed the bus). The DtoH of the raw frame
            # below exists solely so we can draw boxes and write an
            # output MP4; a detection-only deployment drops it.
            frame = cv2.cvtColor(rgb_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
            draw_detections(frame, detections)
            if writer is None:
                fh, fw = frame.shape[:2]
                writer = _make_writer(output_path, info.fps, fw, fh)
            writer.write(frame)

            viddec.ReleaseFrame(packet)

        if packet.bitstream_size <= 0:  # end-of-stream
            break

    if writer is not None:
        writer.release()

    timings.report("pipeline (demux+decode+predict)", output_path)


def process_video_opencv(
    detector: Detector, input_path: str | Path, output_path: str | Path
) -> None:
    """Run the OpenCV CPU-decoding baseline; Step 2 to 4 are identical to the GPU path."""
    cap, info = _open_video(input_path)
    print(f"\nVideo info: {info}")

    writer = _make_writer(output_path, info.fps, info.width, info.height)
    timings = Timings()

    for _ in track(range(info.total_frames), description="Processing video..."):
        # Step 1: Decoding on CPU (OpenCV baseline)
        frame_process_start = time.perf_counter()
        ret, frame = cap.read()  # BGR
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predict_start = time.perf_counter()
        detections = detector.detect_on_gpu(rgb)
        timings.predict_s += time.perf_counter() - predict_start
        timings.pipeline_s += time.perf_counter() - frame_process_start
        timings.frames += 1

        draw_detections(frame, detections)
        writer.write(frame)

    cap.release()
    writer.release()
    timings.report("pipeline (read+predict)", output_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Parse CLI arguments, build the detector, and run the selected decoder path."""
    p = argparse.ArgumentParser(
        description="GPU-resident YOLO video inference on AMD GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--decoder", default="rocdecode", choices=["opencv", "rocdecode"], help="Video decoder backend")
    p.add_argument("--model", default="model.mxr", type=Path, help="MIGraphX .mxr model path")
    p.add_argument("--input", default="data/peloton_sample_ai_gen.mp4", type=Path, help="Input video path (MP4/MKV; H.264/H.265 for rocdecode)")
    p.add_argument("--output", default="output.mp4", type=Path, help="Output video path")
    p.add_argument("--conf-threshold", default=0.25, type=float, help="Detection confidence threshold")
    args = p.parse_args()

    print(f"Loading model from: {args.model}")
    detector = Detector(model_path=args.model, conf_threshold=args.conf_threshold)
    if args.decoder == "rocdecode":
        process_video_rocdecode(detector, args.input, args.output)
    else:
        process_video_opencv(detector, args.input, args.output)


if __name__ == "__main__":
    main()
