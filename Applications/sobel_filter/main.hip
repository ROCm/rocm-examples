// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#endif

// Read .pgm image
bool read_image(const std::string& filepath, int& width, int& height, std::vector<uint8_t>& data)
{
    std::ifstream file(filepath, std::ios::binary);
    if(!file.is_open())
    {
        std::cout << "Error: Failed to open PGM file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);

    if(line != "P5")
    {
        std::cout << "Error: Unsupported PGM format. Use binary P5 format." << std::endl;
        return false;
    }

    // Skip comments
    while(file.peek() == '#')
    {
        std::getline(file, line);
    }

    // Read image dimensions
    int max_val;
    file >> width >> height >> max_val;
    file.ignore(); // Skip the newline character after max_val

    if(max_val != 255)
    {
        std::cout << "Error: Only max_val=255 is supported." << std::endl;
        return false;
    }

    // Read pixel data
    data.resize(width * height);

    file.read(reinterpret_cast<char*>(data.data()), data.size());
    if(file.fail())
    {
        std::cout << "Error: Failed to read pixel data from file." << std::endl;
        return false;
    }

    std::cout << "Image read successfully:" << std::endl;
    std::cout << "\twidth: " << width << ", height: " << height << "." << std::endl;

    return true;
}

__global__ void sobel_filter_kernel(const uint8_t* input, uint8_t* output, int width, int height)
{
    // Calculate the (x, y) position of the current thread in the image
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure that the thread operates within valid image bounds, excluding the edges
    if(x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        // Compute the horizontal gradient (Gx) using the Sobel operator
        int gx = -(input[(y - 1) * width + (x - 1)] + 2 * input[y * width + (x - 1)]
                   + input[(y + 1) * width + (x - 1)])
                 + (input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)]
                    + input[(y + 1) * width + (x + 1)]);

        // Compute the vertical gradient (Gy) using the Sobel operator
        int gy = -(input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x]
                   + input[(y - 1) * width + (x + 1)])
                 + (input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x]
                    + input[(y + 1) * width + (x + 1)]);

        // Compute the gradient magnitude (approximated as the sum of absolute gradients)
        int magnitude = abs(gx) + abs(gy);

        // Clamp the magnitude to the maximum value of 255 (valid for uint8_t)
        magnitude = min(magnitude, 255);

        // Store the computed magnitude in the output image
        output[y * width + x] = static_cast<uint8_t>(magnitude);
    }

    if((x == 0 || x == width - 1) && y >= 0 && y <= height - 1
       || (y == 0 || y == height - 1) && x >= 0 && x <= width - 1)
    {
        output[y * width + x] = 0;
    }
}

void sobel_filter(int                         width,
                  int                         height,
                  const std::vector<uint8_t>& input,
                  std::vector<uint8_t>&       output)
{
    // Calculate the total size of the image data in bytes
    size_t total_size = width * height * sizeof(uint8_t);

    // Declare pointers for device memory
    uint8_t* d_input  = nullptr;
    uint8_t* d_output = nullptr;

    // Allocate memory on the device for the input and output images
    HIP_CHECK(hipMalloc(&d_input, total_size));
    HIP_CHECK(hipMalloc(&d_output, total_size));

    // Copy the input image data from host to device memory
    HIP_CHECK(hipMemcpy(d_input, input.data(), total_size, hipMemcpyHostToDevice));

    // Define the block size (threads per block) and grid size (blocks per grid)
    const dim3 block_dim(16, 16); // 16x16 threads per block
    const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, // Calculate grid dimensions
                        (height + block_dim.y - 1) / block_dim.y);

    // Launch the Sobel filter kernel on the GPU
    sobel_filter_kernel<<<grid_dim, block_dim>>>(d_input, d_output, width, height);

    // Wait for the GPU to complete the kernel execution
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy the processed image data from device back to the host
    HIP_CHECK(hipMemcpy(output.data(), d_output, total_size, hipMemcpyDeviceToHost));

    // Free the allocated device memory to avoid memory leaks
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    // Indicate successful application of the Sobel filter
    std::cout << "Sobel filter applied successfully." << std::endl;
}

void sobel_filter_serial(int                         width,
                         int                         height,
                         const std::vector<uint8_t>& input,
                         std::vector<uint8_t>&       output)
{
    for(int y = 1; y < height - 1; ++y)
    {
        for(int x = 1; x < width - 1; ++x)
        {
            int gx = -(input[(y - 1) * width + (x - 1)] + 2 * input[y * width + (x - 1)]
                       + input[(y + 1) * width + (x - 1)])
                     + (input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)]
                        + input[(y + 1) * width + (x + 1)]);

            int gy = -(input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x]
                       + input[(y - 1) * width + (x + 1)])
                     + (input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x]
                        + input[(y + 1) * width + (x + 1)]);

            int magnitude = abs(gx) + abs(gy);

            magnitude = std::min(magnitude, 255);

            output[y * width + x] = static_cast<uint8_t>(magnitude);
        }
    }

    for(int y = 0; y < height; ++y)
    {
        output[y * width + 0]           = 0;
        output[y * width + (width - 1)] = 0;
    }
    for(int x = 0; x < width; ++x)
    {
        output[0 * width + x]            = 0;
        output[(height - 1) * width + x] = 0;
    }

    std::cout << "Sobel filter (serial) applied successfully." << std::endl;
}

void process_input(GLFWwindow* window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

void framebuffer_size(GLFWwindow* window, int width, int height)
{
    if(!window)
        return;
    glViewport(0, 0, width, height);
}

void render_image(int                         width,
                  int                         height,
                  const std::vector<uint8_t>& original,
                  const std::vector<uint8_t>& processed)
{
    // Clear the color buffer to prepare for rendering
    glClear(GL_COLOR_BUFFER_BIT);

    // Set pixel storage mode for unpacking pixel data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Create and bind textures for rendering
    GLuint texture[2];
    glGenTextures(2, texture);

    // Bind the original image texture
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 width,
                 height,
                 0,
                 GL_RED,
                 GL_UNSIGNED_BYTE,
                 original.data());
    // Linear filtering for resizing window
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Enable 2D texturing
    glEnable(GL_TEXTURE_2D);

    // Render the original image on the left side of the window
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(-1, -1);
    glTexCoord2f(0, 0);
    glVertex2f(-1, 1);
    glTexCoord2f(1, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 1);
    glVertex2f(0, -1);
    glEnd();

    // Bind the processed image texture
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 width,
                 height,
                 0,
                 GL_RED,
                 GL_UNSIGNED_BYTE,
                 processed.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Render the processed image texture on the right side of the window
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(0, -1);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, -1);
    glEnd();

    // Disable 2D texturing and clean up
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(2, texture);
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<bool>("d", "display", false, "Enable display window for results");
    parser.run_and_exit_if_error();

    const bool display = parser.get<bool>("d");

    // Determine the current working directory and construct the input file path
    std::string srcpath = __FILE__;

#ifdef _WIN32
    char separator = srcpath.find_last_of('/') == std::string::npos ? '\\' : '/';

    if(srcpath.find("sobel_filter") == std::string::npos)
    {
        char fullpath[MAX_PATH];
        if(GetFullPathNameA(srcpath.c_str(), MAX_PATH, fullpath, nullptr))
        {
            srcpath = std::string(fullpath);
            srcpath = srcpath.substr(0, srcpath.find_last_of(separator)) + separator
                      + "Applications" + separator + "sobel_filter" + separator + "main.hip";
        }
        else
        {
            std::cerr << "Failed to find image path." << std::endl;
            return -1;
        }
    }
#elif defined(__linux__)
    constexpr char separator = '/';
#endif

    std::string filepath
        = srcpath.substr(0, srcpath.find_last_of(separator)) + separator + "test.pgm";

    int width;
    int height;

    std::vector<uint8_t> data;
    std::vector<uint8_t> data_processed;
    std::vector<uint8_t> data_validation;

    // Read the input image from file
    if(!read_image(filepath, width, height, data))
    {
        std::cerr << "Failed to read image. Exiting." << std::endl;
        return -1;
    }

    data_processed.resize(width * height);
    data_validation.resize(width * height);

    // Apply the Sobel filter to the input image
    std::cout << "Applying Sobel filter..." << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    sobel_filter(width, height, data, data_processed); // Apply Sobel filter

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    // Print the time taken
    std::cout << "Sobel filter applied in " << elapsed.count() << " seconds." << std::endl;

    sobel_filter_serial(width, height, data, data_validation);

    // Compare results with serial results
    bool valid = true;
    for(size_t i = 0; i < data_validation.size(); ++i)
    {
        if(data_validation[i] != data_processed[i])
        {
            std::cerr << "Mismatch at index " << i
                      << ": CPU = " << static_cast<int>(data_validation[i])
                      << ", GPU = " << static_cast<int>(data_processed[i]) << std::endl;
            valid = false;
            break;
        }
    }

    if(valid)
    {
        std::cout << "Validation passed." << std::endl;
    }
    else
    {
        std::cout << "Validation failed" << std::endl;
        return -1;
    }

    if(display)
    {
        // Initialize GLFW for creating a window and OpenGL context
        if(!glfwInit())
        {
            std::cerr << "Error: Failed to initialize GLFW." << std::endl;
            return -1;
        }

        // Create a GLFW window with dimensions matching the image
        GLFWwindow* window
            = glfwCreateWindow(width * 2, height, "Original/ Processed Image", NULL, NULL);
        if(!window)
        {
            std::cerr << "Error: Failed to create GLFW window. Check if OpenGL is supported on "
                         "your system."
                      << std::endl;
            glfwTerminate();
            return -1;
        }

        // Make the OpenGL context current for the created window
        glfwMakeContextCurrent(window);

        // Set the callback function for resizing the framebuffer
        glfwSetFramebufferSizeCallback(window, framebuffer_size);

        // Load OpenGL functions using GLAD
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cerr << "Failed to initialize OpenGL loader!" << std::endl;
            return -1;
        }

        // Main rendering loop
        while(!glfwWindowShouldClose(window))
        {
            // Clear the screen
            glClear(GL_COLOR_BUFFER_BIT);

            // Render the original and processed images side-by-side
            render_image(width, height, data, data_processed);

            // Swap the front and back buffers to display the rendered image
            glfwSwapBuffers(window);

            // Poll for and process user input events
            glfwPollEvents();
            process_input(window);
        }

        // Clean up and terminate GLFW
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    return 0;
}
