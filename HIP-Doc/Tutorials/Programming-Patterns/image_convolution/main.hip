// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "image.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess)                 \
    {                                        \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
        std::exit(EXIT_FAILURE);             \
    }                                        \
}

// [Sphinx HIP 2D convolution kernel start]
__global__ void conv2d(uint8_t *image, uint8_t *output, float *mask,
                       int image_width, int image_height, int channels,
                       int mask_width, int mask_height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= image_width || y >= image_height) {
    return;
  }

  // Process each channel separately
  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    
    for (int i = 0; i < mask_height; i++) {
      for (int j = 0; j < mask_width; j++) {
        // Calculate the coordinate of the pixel to read.
        int image_x = x + j - mask_width / 2;
        int image_y = y + i - mask_height / 2;

        // Do not read outside the image (zero padding).
        if (image_x < 0 || image_x >= image_width || image_y < 0 ||
            image_y >= image_height) {
          continue;
        }

        // Accumulate the value of the pixel.
        int image_index = (image_y * image_width + image_x) * channels + c;
        int mask_index = i * mask_width + j;
        sum += image[image_index] / 255.0f * mask[mask_index];
      }
    }

    int output_index = (y * image_width + x) * channels + c;
    output[output_index] = static_cast<uint8_t>(sum * 255.0f);
  }
}
// [Sphinx HIP 2D convolution kernel end]

int main(int argc, char** argv) {
  // Determine input and output filenames
  const char* input_filename = (argc > 1) ? argv[1] : "test.jpg";
  const char* output_filename = (argc > 2) ? argv[2] : "test_out.jpg";

  // Load the image from disk
  int width, height, channels;
  uint8_t *image = stbi_load(input_filename, &width, &height, &channels, 0);
  if (!image) {
    std::cerr << "Failed to load image: " << input_filename << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Loaded image: " << input_filename << std::endl;
  std::cout << "Dimensions: " << width << " x " << height << std::endl;
  std::cout << "Channels: " << channels << std::endl;

  // Initialize the mask (box blur filter).
  static const int mask_width = 33;
  static const int mask_height = 33;
  std::vector<float> mask(mask_width * mask_height);
  float mask_value = 1.0f / (mask_width * mask_height);
  for (int i = 0; i < mask_width * mask_height; ++i) {
    mask[i] = mask_value;
  }

  std::cout << "Applying " << mask_width << "x" << mask_height 
            << " box blur filter..." << std::endl;

  // Allocate output image buffer on host
  const std::size_t image_size = width * height * channels;
  std::vector<uint8_t> output_image(image_size);

  // Allocate GPU memory and copy data to the GPU.
  uint8_t *d_image, *d_output;
  float *d_mask;
  HIP_CHECK(hipMalloc(&d_image, image_size * sizeof(uint8_t)));
  HIP_CHECK(hipMalloc(&d_output, image_size * sizeof(uint8_t)));
  HIP_CHECK(hipMalloc(&d_mask, mask_width * mask_height * sizeof(float)));
  
  HIP_CHECK(hipMemcpy(d_image, image, image_size * sizeof(uint8_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mask, mask.data(),
                      mask_width * mask_height * sizeof(float),
                      hipMemcpyHostToDevice));

  // Calculate grid size and launch the kernel.
  dim3 block_size(16, 16, 1);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y, 1);

  std::cout << "Launching kernel with grid size: (" 
            << grid_size.x << ", " << grid_size.y << ")" << std::endl;
  std::cout << "Block size: (" << block_size.x << ", " << block_size.y << ")" 
            << std::endl;

  conv2d<<<grid_size, block_size>>>(d_image, d_output, d_mask, width, height,
                                    channels, mask_width, mask_height);
  
  // Check the kernel launch
  HIP_CHECK(hipGetLastError());
  // Check for kernel execution error
  HIP_CHECK(hipDeviceSynchronize());

  // Copy the data back to the host.
  HIP_CHECK(hipMemcpy(output_image.data(), d_output, 
                      image_size * sizeof(uint8_t),
                      hipMemcpyDeviceToHost));

  // Store the image to disk.
  if (stbi_write_jpg(output_filename, width, height, channels, 
                     output_image.data(), 100)) {
    std::cout << "Successfully wrote output to: " << output_filename << std::endl;
  } else {
    std::cerr << "Failed to write output image: " << output_filename << std::endl;
  }

  // Cleanup
  HIP_CHECK(hipFree(d_image));
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipFree(d_mask));
  stbi_image_free(image);

  std::cout << "Execution completed successfully." << std::endl;

  return EXIT_SUCCESS;
}
