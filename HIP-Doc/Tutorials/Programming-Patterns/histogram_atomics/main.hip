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
#include <iomanip>
#include <algorithm>

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

// Number of histogram bins
constexpr int NUM_BINS = 256;

// [Sphinx HIP histogram kernel with atomics start]
__global__ void calculateHistogram(float *imageData, int *histogram, int width,
                                   int height, int channels, int numBins) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate the index for the pixel (Channel-Height-Width layout)
    int idx = (y * width + x) * channels;

    // Calculate brightness by averaging the RGB values
    float brightness = 0.0f;
    for (int c = 0; c < channels; ++c) {
      brightness += imageData[idx + c];
    }
    brightness /= channels;

    // Determine the corresponding bin (assuming brightness range [0,1])
    int bin = static_cast<int>(brightness * (numBins - 1));
    
    // Clamp bin to valid range
    if (bin >= numBins) bin = numBins - 1;
    if (bin < 0) bin = 0;

    // Atomic operation to avoid race conditions
    atomicAdd(&histogram[bin], 1);
  }
}
// [Sphinx HIP histogram kernel with atomics end]

// Helper function to normalize image data to [0, 1] range
void normalizeImageData(const uint8_t *image, float *normalized, 
                        int width, int height, int channels) {
  const int totalPixels = width * height * channels;
  for (int i = 0; i < totalPixels; ++i) {
    normalized[i] = image[i] / 255.0f;
  }
}

// Helper function to print histogram
void printHistogram(const std::vector<int> &histogram, int width, int height) {
  std::cout << "\nHistogram (Brightness Distribution):\n";
  std::cout << std::string(60, '=') << "\n";
  
  int totalPixels = width * height;
  int maxCount = *std::max_element(histogram.begin(), histogram.end());
  
  // Print histogram bars
  const int barWidth = 50;
  for (size_t i = 0; i < histogram.size(); ++i) {
    if (i % 16 == 0) {  // Print every 16th bin for readability
      float percentage = (histogram[i] * 100.0f) / totalPixels;
      int barLength = (histogram[i] * barWidth) / maxCount;
      
      std::cout << "Bin " << std::setw(3) << i << " [" 
                << std::setw(5) << std::fixed << std::setprecision(2) 
                << percentage << "%]: ";
      std::cout << std::string(barLength, '#') << " (" << histogram[i] << ")\n";
    }
  }
  
  std::cout << std::string(60, '=') << "\n";
  
  // Calculate and print statistics
  double meanBrightness = 0.0;
  for (size_t i = 0; i < histogram.size(); ++i) {
    meanBrightness += (i / (double)histogram.size()) * histogram[i];
  }
  meanBrightness /= totalPixels;
  
  std::cout << "\nStatistics:\n";
  std::cout << "Total pixels: " << totalPixels << "\n";
  std::cout << "Mean brightness: " << std::fixed << std::setprecision(3) 
            << meanBrightness << "\n";
  std::cout << "Number of bins: " << histogram.size() << "\n";
}

// Helper function to validate histogram
bool validateHistogram(const std::vector<int> &histogram, int width, int height) {
  int totalCount = 0;
  for (int count : histogram) {
    totalCount += count;
  }
  
  int expectedCount = width * height;
  if (totalCount != expectedCount) {
    std::cerr << "Validation failed! Expected " << expectedCount 
              << " pixels, got " << totalCount << "\n";
    return false;
  }
  
  return true;
}

int main(int argc, char **argv) {
  // Determine input filename
  const char *input_filename = (argc > 1) ? argv[1] : "test_colored.jpg";

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

  // Normalize image data to [0, 1] range
  const std::size_t imageSize = width * height * channels;
  std::vector<float> normalizedImage(imageSize);
  normalizeImageData(image, normalizedImage.data(), width, height, channels);

  // Initialize histogram on host
  std::vector<int> histogram(NUM_BINS, 0);

  std::cout << "\nCalculating histogram with " << NUM_BINS << " bins...\n";

  // Allocate GPU memory
  float *d_imageData;
  int *d_histogram;
  HIP_CHECK(hipMalloc(&d_imageData, imageSize * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_histogram, NUM_BINS * sizeof(int)));

  // Copy data to GPU
  HIP_CHECK(hipMemcpy(d_imageData, normalizedImage.data(), 
                      imageSize * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_histogram, histogram.data(), 
                      NUM_BINS * sizeof(int), hipMemcpyHostToDevice));

  // Calculate grid size and launch the kernel
  dim3 block_size(16, 16, 1);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y, 1);

  std::cout << "Launching kernel with grid size: (" << grid_size.x << ", " 
            << grid_size.y << ")" << std::endl;
  std::cout << "Block size: (" << block_size.x << ", " << block_size.y << ")" 
            << std::endl;

  // Launch kernel
  calculateHistogram<<<grid_size, block_size>>>(
      d_imageData, d_histogram, width, height, channels, NUM_BINS);

  // Check the kernel launch
  HIP_CHECK(hipGetLastError());
  // Check for kernel execution error
  HIP_CHECK(hipDeviceSynchronize());

  // Copy histogram back to host
  HIP_CHECK(hipMemcpy(histogram.data(), d_histogram, 
                      NUM_BINS * sizeof(int), hipMemcpyDeviceToHost));

  // Print and validate histogram
  printHistogram(histogram, width, height);

  if (validateHistogram(histogram, width, height)) {
    std::cout << "\nExecution completed successfully." << std::endl;
  } else {
    std::cerr << "\nExecution failed validation." << std::endl;
  }

  // Cleanup
  HIP_CHECK(hipFree(d_imageData));
  HIP_CHECK(hipFree(d_histogram));
  stbi_image_free(image);

  return EXIT_SUCCESS;
}
