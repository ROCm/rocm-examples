/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocdecode/rocdecode.h>

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"

struct CodecEntry {
  rocDecVideoCodec codec;
  const char *name;
};

struct ChromaEntry {
  rocDecVideoChromaFormat chroma;
  const char *name;
};

struct SurfaceEntry {
  rocDecVideoSurfaceFormat format;
  const char *name;
};

static const CodecEntry codecs[] = {
    {rocDecVideoCodec_MPEG1, "MPEG1"},     {rocDecVideoCodec_MPEG2, "MPEG2"},
    {rocDecVideoCodec_MPEG4, "MPEG4"},     {rocDecVideoCodec_AVC, "AVC/H.264"},
    {rocDecVideoCodec_HEVC, "HEVC/H.265"}, {rocDecVideoCodec_AV1, "AV1"},
    {rocDecVideoCodec_VP8, "VP8"},         {rocDecVideoCodec_VP9, "VP9"},
};

static const ChromaEntry chroma_formats[] = {
    {rocDecVideoChromaFormat_Monochrome, "Monochrome"},
    {rocDecVideoChromaFormat_420, "YUV 4:2:0"},
    {rocDecVideoChromaFormat_422, "YUV 4:2:2"},
    {rocDecVideoChromaFormat_444, "YUV 4:4:4"},
};

static const uint32_t bit_depths[] = {8, 10, 12};

static const SurfaceEntry surface_formats[] = {
    {rocDecVideoSurfaceFormat_NV12, "NV12"},
    {rocDecVideoSurfaceFormat_P016, "P016"},
    {rocDecVideoSurfaceFormat_YUV444, "YUV444"},
    {rocDecVideoSurfaceFormat_YUV444_16Bit, "YUV444_16Bit"},
    {rocDecVideoSurfaceFormat_YUV420, "YUV420"},
    {rocDecVideoSurfaceFormat_YUV420_16Bit, "YUV420_16Bit"},
    {rocDecVideoSurfaceFormat_YUV422, "YUV422"},
    {rocDecVideoSurfaceFormat_YUV422_16Bit, "YUV422_16Bit"},
};

// One row of the output table. Bit depths with otherwise identical
// capabilities are merged into a single row.
struct CapRow {
  std::string chroma;
  std::vector<uint32_t> bit_depths;
  int num_decoders;
  std::string min_res;
  std::string max_res;
  std::string formats;

  bool same_caps(const CapRow &o) const {
    return chroma == o.chroma && num_decoders == o.num_decoders &&
           min_res == o.min_res && max_res == o.max_res && formats == o.formats;
  }
};

std::string bit_depths_string(const std::vector<uint32_t> &depths) {
  std::string s;
  for (uint32_t bd : depths) {
    if (!s.empty()) {
      s += ", ";
    }
    s += std::to_string(bd);
  }
  return s;
}

std::string output_formats_string(uint16_t output_format_mask) {
  std::string formats;
  for (const auto &surf : surface_formats) {
    if (output_format_mask & (1 << surf.format)) {
      if (!formats.empty()) {
        formats += ", ";
      }
      formats += surf.name;
    }
  }
  return formats.empty() ? "none" : formats;
}

int main(int argc, char **argv) {
  // Parse command-line arguments
  cli::Parser parser(argc, argv);
  parser.set_optional<int>(
      "d", "device", 0,
      "GPU device ID (0 for the first device, 1 for the second, etc.)");
  parser.run_and_exit_if_error();

  const int device_id = parser.get<int>("d");
  // RocdecDecodeCaps::device_id is a uint8_t, so cap the valid range at 255.
  if (device_id < 0 || device_id > 255) {
    std::cerr << "Error: invalid device ID " << device_id
              << "; expected an integer in [0, 255]." << std::endl;
    return 1;
  }

  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (device_id >= num_devices) {
    std::cerr << "Error: device ID " << device_id << " is out of range; "
              << num_devices << " device(s) available." << std::endl;
    return 1;
  }

  hipDeviceProp_t hip_dev_prop;
  HIP_CHECK(hipSetDevice(device_id));
  HIP_CHECK(hipGetDeviceProperties(&hip_dev_prop, device_id));

  char pci_bus_id[64] = {0};
  HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id));

  std::cout << "Decoder capabilities for GPU device " << device_id << " - "
            << hip_dev_prop.name << " [" << hip_dev_prop.gcnArchName
            << "] on PCI bus " << pci_bus_id << std::endl
            << std::endl;

  std::cout << std::left << std::setw(12) << "Codec" << std::setw(16)
            << "Chroma" << std::setw(15) << "Bit Depth" << std::setw(9)
            << "Decoders" << std::setw(16) << "Min (WxH)" << std::setw(20)
            << "Max (WxH)" << "Output Surface Formats" << std::endl;
  std::cout << std::string(110, '-') << std::endl;

  int num_supported = 0;
  for (const auto &codec : codecs) {
    // Collect supported (chroma, bit-depth) caps and merge rows that share the
    // same chroma, decoder count, min/max resolution, and output formats so
    // that bit depths with identical capabilities appear on a single row.
    std::vector<CapRow> rows;
    for (const auto &chroma : chroma_formats) {
      for (uint32_t bit_depth : bit_depths) {
        RocdecDecodeCaps decode_caps = {};
        decode_caps.device_id = static_cast<uint8_t>(device_id);
        decode_caps.codec_type = codec.codec;
        decode_caps.chroma_format = chroma.chroma;
        decode_caps.bit_depth_minus_8 = bit_depth - 8;

        rocDecStatus status = rocDecGetDecoderCaps(&decode_caps);
        if (status != ROCDEC_SUCCESS) {
          std::cerr << "Error: rocDecGetDecoderCaps failed: "
                    << rocDecGetErrorName(status) << std::endl;
          return 1;
        }
        if (!decode_caps.is_supported) {
          continue;
        }
        num_supported++;

        CapRow row;
        row.chroma = chroma.name;
        row.bit_depths = {bit_depth};
        row.num_decoders = static_cast<int>(decode_caps.num_decoders);
        row.min_res = std::to_string(decode_caps.min_width) + "x" +
                      std::to_string(decode_caps.min_height);
        row.max_res = std::to_string(decode_caps.max_width) + "x" +
                      std::to_string(decode_caps.max_height);
        row.formats = output_formats_string(decode_caps.output_format_mask);

        bool merged = false;
        for (auto &existing : rows) {
          if (existing.same_caps(row)) {
            existing.bit_depths.push_back(bit_depth);
            merged = true;
            break;
          }
        }
        if (!merged) {
          rows.push_back(row);
        }
      }
    }

    for (const auto &row : rows) {
      std::cout << std::left << std::setw(12) << codec.name << std::setw(16)
                << row.chroma << std::setw(15)
                << (bit_depths_string(row.bit_depths) + "-bit") << std::setw(9)
                << row.num_decoders << std::setw(16) << row.min_res
                << std::setw(20) << row.max_res << row.formats << std::endl;
    }
  }

  std::cout << std::endl
            << "info: " << num_supported
            << " supported codec/chroma/bit-depth combination(s)." << std::endl;
  return 0;
}
