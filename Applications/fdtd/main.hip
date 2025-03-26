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

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// Physical constants.
__device__ __constant__ constexpr float c0 = 299792458.0f; // Speed of light (m/s).
__device__ __constant__ constexpr float mu0
    = 1.256637061e-6; // Vacuum permeability (H/m), 4.0 * m_pi * 1e-7.
__device__ __constant__ constexpr float epsilon0
    = 8.854187819e-12; //Vacuum permittivity (F/m), 1.0 / (mu0 * c0 * c0).
__device__ __constant__ constexpr float m_pi = 3.14159265358979323846;

// Simulation parameters.
struct FDTD_Setup
{
    unsigned int nx, ny, nz; // Grid dimensions.
    float        dx, dy, dz; // Spatial steps (m).
    unsigned int pml_layers; // PML layers at each boundary.
    float        pml_order;
    unsigned int source_x, source_y, source_z; // Source location.
    float        src_freq; // Source center frequency (Hz).
    float        src_amp; // Source amplitude (V/m).
    float        dt; // Time step (s).
    unsigned int size; // Size of field components.

    FDTD_Setup(unsigned int nx_,
               unsigned int ny_,
               unsigned int nz_,
               float        dx_,
               float        dy_,
               float        dz_,
               unsigned int pml_layers_,
               float        pml_order_,
               unsigned int src_x,
               unsigned int src_y,
               unsigned int src_z,
               float        src_freq_,
               float        src_amp_)
        : nx(nx_)
        , ny(ny_)
        , nz(nz_)
        , dx(dx_)
        , dy(dy_)
        , dz(dz_)
        , pml_layers(pml_layers_)
        , pml_order(pml_order_)
        , source_x(src_x)
        , source_y(src_y)
        , source_z(src_z)
        , src_freq(src_freq_)
        , src_amp(src_amp_)
    {
        // Calculate sizes of components
        size = nx * ny * nz;

        // Calculate time step based on CFL condition.
        dt = 0.99f / (c0 * std::sqrt(1.0f / (dx * dx) + 1.0f / (dy * dy) + 1.0f / (dz * dz)));
    }
};

// Medium parameters.
struct Medium_Params
{
    float *mu_r_x, *mu_r_y, *mu_r_z; // Relative permeability.
    float *sigma_m_x, *sigma_m_y, *sigma_m_z; // Magnetic loss parameter.
    float *epsilon_r_x, *epsilon_r_y, *epsilon_r_z; // Relative permittivity.
    float *sigma_x, *sigma_y, *sigma_z; // Electrical conductivity.
};

// Components of fields.
struct Field_Dims
{
    float *Hx, *Hy, *Hz; // Magnetic field components.
    float *Ex, *Ey, *Ez; // Electric field components.
};

// Index transformation for components of fields.
__host__ __device__ __forceinline__ unsigned int
    get_index(const FDTD_Setup& setup, unsigned int x, unsigned int y, unsigned int z)
{
    return x + (y * setup.nx) + (z * setup.nx * setup.ny);
}

// Initialization functions.
void malloc_medium(const FDTD_Setup& setup, Medium_Params& medium);
void customize_medium(const FDTD_Setup& setup, Medium_Params& medium);
void malloc_field(const FDTD_Setup& setup, Field_Dims& data);

void malloc_medium_d(const FDTD_Setup& setup, Medium_Params& medium);
void malloc_field_d(const FDTD_Setup& setup, Field_Dims& data);

void memcpy_medium(const FDTD_Setup&    setup,
                   Medium_Params&       dst,
                   const Medium_Params& src,
                   hipMemcpyKind        kind);
void memcpy_field(const FDTD_Setup& setup,
                  Field_Dims&       dst,
                  const Field_Dims& src,
                  hipMemcpyKind     kind);

// Function to save current state of the fields.
void write_field(const FDTD_Setup& setup, const Field_Dims& data, const fs::path path);

// Function prototype for pulse generation
float get_pulse_amp(FDTD_Setup setup, unsigned int time_step);

// Function prototypes for serial implementation.
void apply_source(FDTD_Setup& setup, float* Ez, float pulse);
void updateH_serial(const FDTD_Setup& setup, Medium_Params& medium, Field_Dims& data);
void updateE_serial(const FDTD_Setup& setup, Medium_Params& medium, Field_Dims& data);

// Functions for serial implementation.
void cleanup_medium(Medium_Params& medium);
void cleanup_field(Field_Dims& data);
void cleanup_medium_d(Medium_Params& medium);
void cleanup_field_d(Field_Dims& data);

// Kernel functions for HIP implementation.
__global__ void apply_source_kernel(FDTD_Setup setup, float* Ez, float pulse);
__global__ void updateH_kernel(FDTD_Setup   setup,
                               float*       H,
                               const float* E1,
                               const float* E2,
                               float*       mu_r,
                               float*       sigma_m_,
                               float        d1,
                               float        d2,
                               unsigned int offset1,
                               unsigned int offset2,
                               unsigned int bound_x,
                               unsigned int bound_y,
                               unsigned int bound_z);

__global__ void updateEx_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ex, const float* Hz, const float* Hy);
__global__ void updateEy_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ey, const float* Hx, const float* Hz);
__global__ void updateEz_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ez, const float* Hy, const float* Hx);

// Function for checking the errors during the calculation on device.
float check_consistency(FDTD_Setup& setup,
                        Field_Dims& data,
                        Field_Dims& data_d,
                        Field_Dims& data_val);

int main(int argc, char* argv[])
{
    // Parse user input.
    cli::Parser parser(argc, argv);
    parser.set_optional<unsigned int>("t", "timestep", 30, "Number of iteration timesteps");
    parser.set_optional<unsigned int>("s", "sample", 3, "Number of steps between two samples");
    parser.set_optional<std::string>("o", "output", "", "Path to the output file");
    parser.run_and_exit_if_error();

    const unsigned int timesteps    = parser.get<unsigned int>("t");
    const unsigned int sample_steps = parser.get<unsigned int>("s");
    const fs::path     output_path  = parser.get<std::string>("o");

    const bool output = parser.doesArgumentExist("o", "output");

    FDTD_Setup setup(128,
                     128,
                     128, // nx, ny, nz.
                     0.01,
                     0.01,
                     0.01, // dx, dy, dz (1cm grid).
                     8, // number of pml layers.
                     3.0f, // order of polynomial pml layers.
                     30,
                     30,
                     30, // source position.
                     2e9, // 2 GHz source.
                     1.0 // source amplitude.
    );

    // Prepare structures for medium parameters and field values both on host (CPU) and device (GPU).
    Medium_Params medium, medium_d; // medium: on host, medium_d: on device
    Field_Dims    data, data_d, data_val;

    // Variables to accumulate computation times for serial CPU vs. HIP GPU implementations.
    float time_track_serial = 0.0f, time_track_hip = 0.0f;

    // Allocate host memory for medium parameters and field arrays.
    malloc_medium(setup, medium);
    malloc_field(setup, data);
    malloc_field(setup, data_val);

    // Allocate device (GPU) memory for medium parameters and field arrays.
    malloc_medium_d(setup, medium_d);
    malloc_field_d(setup, data_d);

    // Customize the medium (e.g., set up PML regions, any objects in the simulation, etc.).
    // Then copy the initialized medium parameters from host to device.
    customize_medium(setup, medium);
    memcpy_medium(setup, medium_d, medium, hipMemcpyHostToDevice);

    // Define the block and grid dimensions for GPU kernels.
    // block(4, 4, 4) means each block has 4x4x4 = 64 threads;
    dim3 block(4, 4, 4);
    dim3 grid((setup.nx + block.x - 1) / block.x,
              (setup.ny + block.y - 1) / block.y,
              (setup.nz + block.z - 1) / block.z);

    hipEvent_t event_H_done;
    HIP_CHECK(hipEventCreate(&event_H_done));

    // Create separate HIP streams to enable concurrent execution of partial updates.
    // We use 3 streams here, one for each dimension of the field updates.
    const int   dimensions = 3;
    hipStream_t streams[dimensions];
    for(int i = 0; i < dimensions; i++)
    {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // Main FDTD time-stepping loop.
    for(unsigned int i = 0; i < timesteps; i++)
    {
        float pulse = get_pulse_amp(setup, i);
        // 1) CPU SERIAL IMPLEMENTATION
        // Apply the source (e.g., a sine wave) at the specified grid location in the Ez field.
        apply_source(setup, data_val.Ez, pulse);

        // Record the start time, run the serial H and E updates, then record the end time.
        auto record_start = std::chrono::high_resolution_clock::now();
        // Update magnetic fields (Hx, Hy, Hz) on CPU
        updateH_serial(setup, medium, data_val);
        // Update electric fields (Ex, Ey, Ez) on CPU
        updateE_serial(setup, medium, data_val);
        auto record_end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time for this iteration on the CPU.
        std::chrono::duration<float> elapsed = record_end - record_start;

        time_track_serial += elapsed.count();

        // 2) GPU IMPLEMENTATION
        // First, add the same source on the device side (GPU) via a kernel call.
        // We'll run this kernel with a single block and a single thread, synchronizing afterward.
        apply_source_kernel<<<1, 1, 0, streams[0]>>>(setup, data_d.Ez, pulse);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipStreamSynchronize(streams[0]));

        // Record the start time for GPU operations (for H updates).
        record_start = std::chrono::high_resolution_clock::now();

        // Launch kernels to update Hx, Hy, Hz in parallel on different streams.
        updateH_kernel<<<grid, block, 0, streams[0]>>>(setup,
                                                       data_d.Hx,
                                                       data_d.Ez,
                                                       data_d.Ey,
                                                       medium_d.mu_r_x,
                                                       medium_d.sigma_m_x,
                                                       setup.dz,
                                                       setup.dy,
                                                       setup.nx,
                                                       setup.nx * setup.ny,
                                                       setup.nx,
                                                       setup.ny - 1,
                                                       setup.nz - 1);
        HIP_CHECK(hipPeekAtLastError());
        updateH_kernel<<<grid, block, 0, streams[1]>>>(setup,
                                                       data_d.Hy,
                                                       data_d.Ex,
                                                       data_d.Ez,
                                                       medium_d.mu_r_y,
                                                       medium_d.sigma_m_y,
                                                       setup.dx,
                                                       setup.dz,
                                                       setup.nx * setup.ny,
                                                       1,
                                                       setup.nx - 1,
                                                       setup.ny,
                                                       setup.nz - 1);
        HIP_CHECK(hipPeekAtLastError());
        updateH_kernel<<<grid, block, 0, streams[2]>>>(setup,
                                                       data_d.Hz,
                                                       data_d.Ey,
                                                       data_d.Ex,
                                                       medium_d.mu_r_z,
                                                       medium_d.sigma_m_z,
                                                       setup.dy,
                                                       setup.dx,
                                                       1,
                                                       setup.nx,
                                                       setup.nx - 1,
                                                       setup.ny - 1,
                                                       setup.nz);
        HIP_CHECK(hipPeekAtLastError());

        HIP_CHECK(hipEventRecord(event_H_done, streams[0]));

        // Synchronize the three streams to ensure H-field updates are finished before E-field updates.
        HIP_CHECK(hipStreamSynchronize(streams[0]));
        HIP_CHECK(hipStreamSynchronize(streams[1]));
        HIP_CHECK(hipStreamSynchronize(streams[2]));

        // Now update Ex, Ey, Ez on the GPU in parallel streams.
        updateEx_kernel<<<grid, block, 0, streams[0]>>>(setup,
                                                        medium_d,
                                                        data_d.Ex,
                                                        data_d.Hz,
                                                        data_d.Hy);
        HIP_CHECK(hipPeekAtLastError());
        updateEy_kernel<<<grid, block, 0, streams[1]>>>(setup,
                                                        medium_d,
                                                        data_d.Ey,
                                                        data_d.Hx,
                                                        data_d.Hz);
        HIP_CHECK(hipPeekAtLastError());
        updateEz_kernel<<<grid, block, 0, streams[2]>>>(setup,
                                                        medium_d,
                                                        data_d.Ez,
                                                        data_d.Hy,
                                                        data_d.Hx);
        HIP_CHECK(hipPeekAtLastError());

        // Record the end time (still need a final sync).
        record_end = std::chrono::high_resolution_clock::now();

        elapsed = record_end - record_start;
        time_track_hip += elapsed.count();

        // Synchronize again to ensure the electric field updates have completed.
        HIP_CHECK(hipStreamSynchronize(streams[0]));
        HIP_CHECK(hipStreamSynchronize(streams[1]));
        HIP_CHECK(hipStreamSynchronize(streams[2]));

        // Every 'sample_steps' iterations, we output the current iteration,
        // the accumulated serial time, HIP time, and a consistency check (relative error).
        if((i + 1) % sample_steps == 0)
        {
            std::cout << "iteration: " << i + 1 << "\t time serial: " << time_track_serial
                      << "\t time HIP: " << time_track_hip << "\t consistency: "
                      << check_consistency(setup, data, data_d, data_val) * 100 << "%" << std::endl;

            if(output)
            {
                const fs::path path = output_path / ("fdtd_step_" + std::to_string(i));

                if(!fs::exists(path))
                {
                    fs::create_directories(path);
                }

                write_field(setup, data, path);
            }
        }
    }

    // Clean up host arrays for field data and medium parameters.
    cleanup_field(data);
    cleanup_field(data_val);
    cleanup_medium(medium);

    // Clean up device arrays for field data and medium parameters.
    cleanup_field_d(data_d);
    cleanup_medium_d(medium_d);

    // Exit the program successfully.
    return 0;
}

float get_pulse_amp(FDTD_Setup setup, unsigned int time_step)
{
    // Calculate the current time based on the iteration idx.
    const float t = time_step * setup.dt;

    // Generate the sinusoidal pulse according to the source frequency.
    const float pulse = sin(2.0f * m_pi * setup.src_freq * t);

    return setup.src_amp * pulse;
}

// GPU kernel to apply the source (e.g., a sinusoidal wave) to the Ez field.
// This kernel injects the excitation only if we are at threadIdx.x=0 and blockIdx.x=0,
// effectively applying the source at a single point in the domain.
__global__ void apply_source_kernel(FDTD_Setup setup, float* Ez, float pulse)
{
    // We only need one thread to add the source, so we check if it is the "first" thread.
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Compute the idx of the Ez field array where the source will be injected.
        const unsigned int idx = get_index(setup, setup.source_x, setup.source_y, setup.source_z);

        // Add the sinusoidal pulse into the Ez component at the specified source location.
        Ez[idx] += pulse;
    }
}

// This kernel implements the finite-difference update equation for Hx in the Yee scheme.
__global__ void updateHx_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Hx, const float* Ez, const float* Ey)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x >= setup.nx || y >= setup.ny - 1 || z >= setup.nz - 1)
    {
        return;
    }

    const unsigned int idx = get_index(setup, x, y, z);

    const float dy = setup.dy, dz = setup.dz, dt = setup.dt;

    const float ez2 = Ez[get_index(setup, x, y + 1, z)];
    const float ez1 = Ez[get_index(setup, x, y, z)];
    const float ey2 = Ey[get_index(setup, x, y, z + 1)];
    const float ey1 = Ey[get_index(setup, x, y, z)];

    float mu = medium.mu_r_x[idx] * mu0; // Effective permeability = mu_r * mu0

    float sigma_m = medium.sigma_m_x[idx]; // Magnetic conductivity / loss

    // Compute the curl of E in the direction appropriate for updating Hx:
    //    curl_x(E) = (∂Ez/∂y) - (∂Ey/∂z).
    float curl = (ez2 - ez1) / dy - (ey2 - ey1) / dz;

    // Compute the auxiliary factor for magnetic conductivity (used in a Crank-Nicolson-like update).
    // loss_coeff = (σm * Δt) / (2 * μ).
    float loss_coeff = sigma_m * dt / (2 * mu);

    // Update Hx using the standard FDTD formula (with correction for magnetic loss):
    //    Hx_new = [ (1 - loss_coeff) * Hx_old - (Δt/μ) * curl ] / (1 + loss_coeff ).
    // This takes into account the PML or dispersive medium losses via sigma_m.
    Hx[idx] = ((1.0f - loss_coeff) * Hx[idx] - (curl / mu * dt)) / (1.0f + loss_coeff);
}

__global__ void updateH_kernel(FDTD_Setup   setup,
                               float*       H,
                               const float* E1,
                               const float* E2,
                               float*       mu_r,
                               float*       sigma_m_,
                               float        d1,
                               float        d2,
                               unsigned int offset1,
                               unsigned int offset2,
                               unsigned int bound_x,
                               unsigned int bound_y,
                               unsigned int bound_z)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x >= bound_x || y >= bound_y || z >= bound_z)
    {
        return;
    }

    const unsigned int idx = get_index(setup, x, y, z);

    const float dt = setup.dt;

    const float e1_1 = E1[idx + offset1];
    const float e1_0 = E1[idx];
    const float e2_1 = E2[idx + offset2];
    const float e2_0 = E2[idx];

    float curl, mu, sigma_m, loss_coeff;

    mu      = mu_r[idx] * mu0;
    sigma_m = sigma_m_[idx];

    curl = (e1_1 - e1_0) / d2 - (e2_1 - e2_0) / d1;

    loss_coeff = sigma_m * dt / (2 * mu);

    H[idx] = ((1.0f - loss_coeff) * H[idx] - (curl / mu * dt)) / (1.0f + loss_coeff);
}

// This kernel computes Ex at each valid (x, y, z) position on the grid. The update formula
// includes the curl of the magnetic field (Hz and Hy) and accounts for electrical conductivity
// (sigma) and permittivity (epsilon) to handle lossy media or PML regions.
__global__ void updateEx_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ex, const float* Hz, const float* Hy)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    if(x >= nx - 1 || y >= ny || z >= nz)
    {
        return;
    }

    const unsigned int idx = get_index(setup, x, y, z);

    // Fetch spatial step sizes and time step from the setup structure.
    const float dy = setup.dy, dz = setup.dz, dt = setup.dt;

    unsigned int iy, iz;

    float hz2, hz1, hy2, hy1;
    float curl, epsilon, sigma, loss_coeff;

    // For Ex, the relevant components of the curl come from Hz and Hy:
    //   curl_x(H) = ( ∂Hz/∂y ) - ( ∂Hy/∂z ).
    // We must carefully pick the correct Hz/Hy values at staggered grid positions
    // around the point (x, y, z). Below, we handle the boundaries by choosing
    // adjacent indices.
    iy  = max(0, static_cast<int>(y) - 1);
    hz1 = Hz[get_index(setup, x, iy, z)];

    iy  = min(ny - 2, y);
    hz2 = Hz[get_index(setup, x, y, z)];

    iz  = max(0, static_cast<int>(z) - 1);
    hy1 = Hy[get_index(setup, x, y, iz)];

    iz  = min(nz - 2, z);
    hy2 = Hy[get_index(setup, x, y, iz)];

    // Retrieve the permittivity (epsilon) and conductivity (sigma) for this point.
    // epsilon = medium.epsilon_r_x * epsilon0, where epsilon_r_x is the relative permittivity in x direction.
    epsilon = medium.epsilon_r_x[idx] * epsilon0;
    sigma   = medium.sigma_x[idx];

    // Compute the curl_x(H) = (Hz2 - Hz1)/dy - (Hy2 - Hy1)/dz.
    // The differences reflect partial derivatives in y and z directions.
    curl = (hz2 - hz1) / dy - (hy2 - hy1) / dz;

    // Calculate the conductivity-related factor:
    // loss_coeff = (sigma * dt) / (2 * epsilon).
    // This accounts for losses or PML damping in the electric field update.
    loss_coeff = sigma * dt / (2.0f * epsilon);

    // Update Ex using the FDTD formula with conductivity:
    //   Ex_new = [ (1 - loss_coeff)*Ex_old + (dt/epsilon)*curl ] / (1 + loss_coeff ).
    // This form integrates the conduction current in a manner akin to a Crank-Nicolson approach for lossy media.
    Ex[idx] = ((1.0f - loss_coeff) * Ex[idx] + (curl / epsilon * dt)) / (1.0f + loss_coeff);
}

// This kernel implements the finite-difference update equation for Ey in the Yee scheme.
__global__ void updateEy_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ey, const float* Hx, const float* Hz)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    if(x >= nx || y >= ny - 1 || z >= nz)
    {
        return;
    }

    const unsigned int idx = get_index(setup, x, y, z);

    const float dx = setup.dx, dz = setup.dz, dt = setup.dt;

    unsigned int iz, ix;

    float hx2, hx1, hz2, hz1;
    float curl, epsilon, sigma, loss_coeff;

    epsilon = medium.epsilon_r_y[idx] * epsilon0;
    sigma   = medium.sigma_y[idx];

    iz  = max(0, static_cast<int>(z) - 1);
    hx1 = Hx[get_index(setup, x, y, iz)];

    iz  = min(nz - 2, z);
    hx2 = Hx[get_index(setup, x, y, iz)];

    ix  = max(0, static_cast<int>(x) - 1);
    hz1 = Hz[get_index(setup, ix, y, z)];

    ix  = min(nx - 2, x);
    hz2 = Hz[get_index(setup, ix, y, z)];

    curl = (hx2 - hx1) / dz - (hz2 - hz1) / dx;

    loss_coeff = sigma * dt / (2 * epsilon);

    Ey[idx] = ((1.0f - loss_coeff) * Ey[idx] + (curl / epsilon * dt)) / (1.0f + loss_coeff);
}

// This kernel implements the finite-difference update equation for Ez in the Yee scheme.
__global__ void updateEz_kernel(
    FDTD_Setup setup, Medium_Params medium, float* Ez, const float* Hy, const float* Hx)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    if(x >= nx || y >= ny || z >= nz - 1)
    {
        return;
    }

    const unsigned int idx = get_index(setup, x, y, z);

    const float dx = setup.dx, dy = setup.dz, dt = setup.dt;

    unsigned int ix, iy;

    float hy2, hy1, hx2, hx1;
    float curl, epsilon, sigma, loss_coeff;

    epsilon = medium.epsilon_r_z[idx] * epsilon0;
    sigma   = medium.sigma_z[idx];

    ix  = max(0, static_cast<int>(x) - 1);
    hy1 = Hy[get_index(setup, ix, y, z)];

    ix  = min(nx - 2, x);
    hy2 = Hy[get_index(setup, ix, y, z)];

    iy  = max(0, static_cast<int>(y) - 1);
    hx1 = Hx[get_index(setup, x, iy, z)];

    iy  = min(ny - 2, y);
    hx2 = Hx[get_index(setup, x, iy, z)];

    curl = (hy2 - hy1) / dx - (hx2 - hx1) / dy;

    loss_coeff = sigma * dt / (2 * epsilon);

    Ez[idx] = ((1.0f - loss_coeff) * Ez[idx] + (curl / epsilon * dt)) / (1.0f + loss_coeff);
}

// CPU function to apply the source (e.g., a sinusoidal wave) to the Ez field.
// This is a serial equivalent for the GPU version, used for CPU-side calculations.
void apply_source(FDTD_Setup& setup, float* Ez, float pulse)
{
    // Retrieve the source location from the setup structure.
    const unsigned int idx = get_index(setup, setup.source_x, setup.source_y, setup.source_z);

    // Inject the source (pulse) into the Ez field at the specified grid location.
    Ez[idx] += pulse;
}

void updateH_component(const FDTD_Setup& setup,
                       float*            H,
                       const float*      E1,
                       const float*      E2,
                       float*            mu_r,
                       float*            sigma_m_,
                       float             d1,
                       float             d2,
                       unsigned int      offset1,
                       unsigned int      offset2,
                       unsigned int      bound_x,
                       unsigned int      bound_y,
                       unsigned int      bound_z)
{
    const float dt = setup.dt;

    unsigned int idx;

    float e1_1, e1_0, e2_1, e2_0;
    float curl, mu, sigma_m, loss_coeff;

    // Update Hx in a 3D nested loop: z in [0..nz-2], y in [0..ny-2], x in [0..nx-1].
    for(unsigned int z = 0; z < bound_z; z++)
    {
        for(unsigned int y = 0; y < bound_y; y++)
        {
            for(unsigned int x = 0; x < bound_x; x++)
            {
                idx = get_index(setup, x, y, z);

                // Read required Ez and Ey values to compute the curl component for Hx.
                e1_1 = E1[idx + offset1];
                e1_0 = E1[idx];
                e2_1 = E2[idx + offset2];
                e2_0 = E2[idx];

                // Extract effective permeability and magnetic conductivity.

                mu      = mu_r[idx] * mu0;
                sigma_m = sigma_m_[idx];

                // Compute curl_x(E) = (∂Ez/∂y) - (∂Ey/∂z).
                curl = (e1_1 - e1_0) / d2 - (e2_1 - e2_0) / d1;

                loss_coeff = sigma_m * dt / (2 * mu);

                // Compute temporary factor for magnetic conductivity:
                // loss_coeff = (σm * Δt) / (2 * μ).

                // Update Hx using FDTD formula with magnetic loss term:
                // Hx_new = [ (1 - loss_coeff)*Hx_old - (Δt/μ) * curl ] / [1 + loss_coeff].
                H[idx] = ((1.0f - loss_coeff) * H[idx] - (curl / mu * dt)) / (1.0f + loss_coeff);
            }
        }
    }
}

// Serial implementation for updating magnetic field
void updateH_serial(const FDTD_Setup& setup, Medium_Params& medium, Field_Dims& data)
{
    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    const float dx = setup.dx, dy = setup.dy, dz = setup.dz;

    updateH_component(setup,
                      data.Hx,
                      data.Ez,
                      data.Ey,
                      medium.mu_r_x,
                      medium.sigma_m_x,
                      dz,
                      dy,
                      nx,
                      nx * ny,
                      nx,
                      ny - 1,
                      nz - 1);

    updateH_component(setup,
                      data.Hy,
                      data.Ex,
                      data.Ez,
                      medium.mu_r_y,
                      medium.sigma_m_y,
                      dx,
                      dz,
                      nx * ny,
                      1,
                      nx - 1,
                      ny,
                      nz - 1);

    updateH_component(setup,
                      data.Hz,
                      data.Ey,
                      data.Ex,
                      medium.mu_r_z,
                      medium.sigma_m_z,
                      dy,
                      dx,
                      1,
                      nx,
                      nx - 1,
                      ny - 1,
                      nz);
}

// Serial implementation for updating electric field
void updateE_serial(const FDTD_Setup& setup, Medium_Params& medium, Field_Dims& data)
{
    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    const float dx = setup.dx, dy = setup.dy, dz = setup.dz, dt = setup.dt;

    float hx1, hx2, hy1, hy2, hz1, hz2;
    float curl, epsilon, sigma, loss_coeff;

    unsigned int ix, iy, iz, idx;

    for(unsigned int z = 0; z < nz; z++)
    {
        for(unsigned int y = 0; y < ny; y++)
        {
            for(unsigned int x = 0; x < nx - 1; x++)
            {
                idx     = get_index(setup, x, y, z);
                epsilon = medium.epsilon_r_x[idx] * epsilon0;
                sigma   = medium.sigma_x[idx];

                iy  = std::max(0, static_cast<int>(y) - 1);
                hz1 = data.Hz[get_index(setup, x, iy, z)];

                iy  = std::min(ny - 2, y);
                hz2 = data.Hz[get_index(setup, x, y, z)];

                iz  = std::max(0, static_cast<int>(z) - 1);
                hy1 = data.Hy[get_index(setup, x, y, iz)];

                iz  = std::min(nz - 2, z);
                hy2 = data.Hy[get_index(setup, x, y, iz)];

                // Calculate the curl component:
                //   curl_x(H) = (Hz2 - Hz1)/dy - (Hy2 - Hy1)/dz.
                curl = (hz2 - hz1) / dy - (hy2 - hy1) / dz;

                // Calculate conductivity damping factor for the electric field:
                loss_coeff = sigma * dt / (2.0f * epsilon);

                // Update Ex with the FDTD formula incorporating material losses:
                //   Ex_new = [ (1 - loss_coeff)*Ex_old + (dt/epsilon)*curl_x(H) ] / (1 + loss_coeff ).
                data.Ex[idx]
                    = ((1 - loss_coeff) * data.Ex[idx] + (curl / epsilon * dt)) / (1 + loss_coeff);
            }
        }
    }

    for(unsigned int z = 0; z < nz; z++)
    {
        for(unsigned int y = 0; y < ny - 1; y++)
        {
            for(unsigned int x = 0; x < nx; x++)
            {
                idx     = get_index(setup, x, y, z);
                epsilon = medium.epsilon_r_y[idx] * epsilon0;
                sigma   = medium.sigma_y[idx];

                iz  = std::max(0, static_cast<int>(z) - 1);
                hx1 = data.Hx[get_index(setup, x, y, iz)];

                iz  = std::min(nz - 2, z);
                hx2 = data.Hx[get_index(setup, x, y, iz)];

                ix  = std::max(0, static_cast<int>(x) - 1);
                hz1 = data.Hz[get_index(setup, ix, y, z)];

                ix  = std::min(nx - 2, x);
                hz2 = data.Hz[get_index(setup, ix, y, z)];

                curl = (hx2 - hx1) / dz - (hz2 - hz1) / dx;

                loss_coeff = sigma * dt / (2 * epsilon);

                data.Ey[idx]
                    = ((1 - loss_coeff) * data.Ey[idx] + (curl / epsilon * dt)) / (1 + loss_coeff);
            }
        }
    }

    for(unsigned int z = 0; z < nz - 1; z++)
    {
        for(unsigned int y = 0; y < ny; y++)
        {
            for(unsigned int x = 0; x < nx; x++)
            {
                idx     = get_index(setup, x, y, z);
                epsilon = medium.epsilon_r_z[idx] * epsilon0;
                sigma   = medium.sigma_z[idx];

                ix  = std::max(0, static_cast<int>(x) - 1);
                hy1 = data.Hy[get_index(setup, ix, y, z)];

                ix  = std::min(nx - 2, x);
                hy2 = data.Hy[get_index(setup, ix, y, z)];

                iy  = std::max(0, static_cast<int>(y) - 1);
                hx1 = data.Hx[get_index(setup, x, iy, z)];

                iy  = std::min(ny - 2, y);
                hx2 = data.Hx[get_index(setup, x, iy, z)];

                curl = (hy2 - hy1) / dx - (hx2 - hx1) / dy;

                loss_coeff = sigma * dt / (2 * epsilon);

                data.Ez[idx]
                    = ((1 - loss_coeff) * data.Ez[idx] + (curl / epsilon * dt)) / (1 + loss_coeff);
            }
        }
    }
}

// Define a local lambda function to calculate a polynomial-based conductivity distribution for PML.
//   dist_x, dist_y, dist_z: the distance (in cells) from the interior to the boundary
//   sigma_max_val: the maximum conductivity to reach at the outermost boundary cell
//   pml_layers: total PML layer count
//   pml_order: controls how quickly the conductivity ramps up
inline float polynomial_sigma(float        dist_x,
                              float        dist_y,
                              float        dist_z,
                              float        sigma_max_val,
                              unsigned int pml_layers,
                              float        pml_order)
{
    dist_x = dist_x > 0 ? dist_x : 0;
    dist_y = dist_y > 0 ? dist_y : 0;
    dist_z = dist_z > 0 ? dist_z : 0;
    return sigma_max_val
           * (std::pow((dist_x / pml_layers), pml_order)
              + std::pow((dist_y / pml_layers), pml_order)
              + std::pow((dist_z / pml_layers), pml_order));
};

void customize_H_medium_component(const FDTD_Setup& setup,
                                  float*            mu_r,
                                  float*            sigma_m_,
                                  unsigned int      bound_x,
                                  unsigned int      bound_y,
                                  unsigned int      bound_z)
{
    const float pml_layers = setup.pml_layers, pml_order = setup.pml_order;

    const float sigma_m_max = 1e6f; // Max magnetic conductivity

    unsigned int idx;

    // 1) Magnetic properties for Hx
    //    We loop over valid indices of Hx in a Yee grid: x in [0, nx], y in [0, ny-1], z in [0, nz-1].
    //    By default, set mu_r = 1.0 and sigma_m = 0.0 (vacuum). Then apply PML modifications if in boundary regions.
    for(unsigned int z = 0; z < bound_z; z++)
    {
        for(unsigned int y = 0; y < bound_y; y++)
        {
            for(unsigned int x = 0; x < bound_x; x++)
            {
                // Flatten 3D idx to 1D for the Hx array.
                idx = get_index(setup, x, y, z);

                // Default: vacuum-like (relative mu = 1, no magnetic conductivity).
                mu_r[idx]     = 1.0f;
                sigma_m_[idx] = 0.0f;

                // If we're within the PML region at the low boundary (x or y or z close to 0):
                if(x < pml_layers || y < pml_layers || z < pml_layers)
                {
                    sigma_m_[idx] = polynomial_sigma(pml_layers - x,
                                                     pml_layers - y,
                                                     pml_layers - z,
                                                     sigma_m_max,
                                                     pml_layers,
                                                     pml_order);
                }

                // If we're within the PML region at the high boundary (x or y or z close to domain max):
                //   For Hx, y goes up to ny-1, z up to nz-1, so we adjust the offsets accordingly.
                if(x >= bound_x - pml_layers || y >= bound_y - pml_layers
                   || z >= bound_z - pml_layers)
                {
                    sigma_m_[idx] = polynomial_sigma(x + pml_layers - bound_x,
                                                     y + pml_layers - bound_y,
                                                     z + pml_layers - bound_z,
                                                     sigma_m_max,
                                                     pml_layers,
                                                     pml_order);
                }
            }
        }
    }
}

void customize_E_medium_component(const FDTD_Setup& setup,
                                  float*            epsilon_r,
                                  float*            sigma_,
                                  unsigned int      bound_x,
                                  unsigned int      bound_y,
                                  unsigned int      bound_z)
{
    const float pml_layers = setup.pml_layers, pml_order = setup.pml_order;

    const float sigma_max = 1e6f; // Max electric conductivity

    unsigned int idx;

    // Electric properties for Ex
    //    Ex is defined at x in [0, nx-1], y in [0, ny], z in [0, nz].
    //    Also set a metal region ([60,80) in x,y,z) with large sigma.
    for(unsigned int z = 0; z < bound_z; z++)
    {
        for(unsigned int y = 0; y < bound_y; y++)
        {
            for(unsigned int x = 0; x < bound_x; x++)
            {
                idx = get_index(setup, x, y, z);

                // By default, set vacuum permittivity (relative epsilon = 1.0).
                epsilon_r[idx] = 1.0f;

                // If within the metallic cube, apply maximum electric conductivity.
                if(x >= 60 && x < 80 && y >= 60 && y < 80 && z >= 60 && z < 80)
                {
                    sigma_[idx] = sigma_max;
                }
                else
                {
                    sigma_[idx] = 0.0f;
                }

                // Check PML near lower boundary.
                if(x < pml_layers || y < pml_layers || z < pml_layers)
                {
                    sigma_[idx] = polynomial_sigma(pml_layers - x,
                                                   pml_layers - y,
                                                   pml_layers - z,
                                                   sigma_max,
                                                   pml_layers,
                                                   pml_order);
                }

                // Check PML near upper boundary.
                // For Ex, x goes up to nx-1, y up to ny, z up to nz, so we adjust offsets accordingly.
                if(x >= bound_x - pml_layers || y >= bound_y - pml_layers
                   || z >= bound_z - pml_layers)
                {
                    sigma_[idx] = polynomial_sigma(x + pml_layers - bound_x,
                                                   y + pml_layers - bound_y,
                                                   z + pml_layers - bound_z,
                                                   sigma_max,
                                                   pml_layers,
                                                   pml_order);
                }
            }
        }
    }
}

/* Customizes the medium parameters by setting material properties for each grid point,
 *        including PML (Perfectly Matched Layers) and a metallic region.
 *
 * This function modifies the Medium_Params structure to account for:
 *   - A vacuum default (relative permeability = 1, relative permittivity = 1, zero conductivity).
 *   - A non-magnetic metal cube in the region [60, 80) for x, y, and z.
 *   - PML (Perfectly Matched Layer) boundaries around the edges of the computational domain,
 *     adding gradually increasing conductivity and/or magnetic loss terms near the domain boundaries.
*/
void customize_medium(const FDTD_Setup& setup, Medium_Params& medium)
{
    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;

    // 1) Magnetic properties for Hx
    customize_H_medium_component(setup, medium.mu_r_x, medium.sigma_m_x, nx, ny - 1, nz - 1);

    // 2) Magnetic properties for Hy
    customize_H_medium_component(setup, medium.mu_r_y, medium.sigma_m_y, nx - 1, ny, nz - 1);

    // 3) Magnetic properties for Hz
    customize_H_medium_component(setup, medium.mu_r_z, medium.sigma_m_z, nx - 1, ny - 1, nz);

    // 4) Electric properties for Ex
    customize_E_medium_component(setup, medium.epsilon_r_x, medium.sigma_x, nx - 1, ny, nz);

    // 5) Electric properties for Ey
    customize_E_medium_component(setup, medium.epsilon_r_y, medium.sigma_y, nx, ny - 1, nz);

    // 6) Electric properties for Ez
    customize_E_medium_component(setup, medium.epsilon_r_z, medium.sigma_z, nx, ny, nz - 1);
}

float check_cell_consistency(float n, float n_val, float tolerance)
{
    float error, error_percent;

    error = std::abs(n - n_val);

    if(error < tolerance)
    {
        error_percent = 0.0f;
    }
    else if(n_val == 0)
    {
        error_percent = 1.0f;
    }
    else
    {
        error_percent = std::min(error / std::abs(n_val), 1.0f);
    }
    return error_percent;
}

float check_consistency(FDTD_Setup& setup,
                        Field_Dims& data,
                        Field_Dims& data_d,
                        Field_Dims& data_val)
{
    memcpy_field(setup, data, data_d, hipMemcpyDeviceToHost);

    const unsigned int x = setup.source_x, y = setup.source_y, z = setup.source_z;

    unsigned int ihx, ihy, ihz, iex, iey, iez;

    float error = 0, tolerance = std::abs(data_val.Ez[get_index(setup, x, y, z)] * 1e-9);

    // The recorded points are sampled around the source.
    for(int k = -1; k < 2; k++)
    {
        for(int j = -1; j < 2; j++)
        {
            for(int i = -1; i < 2; i++)
            {
                ihx = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);
                ihy = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);
                ihz = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);
                iex = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);
                iey = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);
                iez = get_index(setup, i * 10 + x, j * 10 + y, k * 10 + z);

                // The error is calculated by percentage of difference between host and device result
                // compared to host result.
                error += check_cell_consistency(data.Hx[ihx], data_val.Hx[ihx], tolerance);
                error += check_cell_consistency(data.Hy[ihy], data_val.Hy[ihy], tolerance);
                error += check_cell_consistency(data.Hz[ihz], data_val.Hz[ihz], tolerance);
                error += check_cell_consistency(data.Ex[iex], data_val.Ex[iex], tolerance);
                error += check_cell_consistency(data.Ey[iey], data_val.Ey[iey], tolerance);
                error += check_cell_consistency(data.Ez[iez], data_val.Ez[iez], tolerance);
            }
        }
    }

    // The consistency is represented in percentage.
    return 1.0f - (error / (3 * 3 * 3 * 6.0f));
}

// Allocate the memory for medium
void malloc_medium(const FDTD_Setup& setup, Medium_Params& medium)
{
    medium.mu_r_x      = new float[setup.size]();
    medium.mu_r_y      = new float[setup.size]();
    medium.mu_r_z      = new float[setup.size]();
    medium.sigma_m_x   = new float[setup.size]();
    medium.sigma_m_y   = new float[setup.size]();
    medium.sigma_m_z   = new float[setup.size]();
    medium.epsilon_r_x = new float[setup.size]();
    medium.epsilon_r_y = new float[setup.size]();
    medium.epsilon_r_z = new float[setup.size]();
    medium.sigma_x     = new float[setup.size]();
    medium.sigma_y     = new float[setup.size]();
    medium.sigma_z     = new float[setup.size]();
}

// Allocate the memory for field components.
void malloc_field(const FDTD_Setup& setup, Field_Dims& data)
{
    data.Hx = new float[setup.size]();
    data.Hy = new float[setup.size]();
    data.Hz = new float[setup.size]();
    data.Ex = new float[setup.size]();
    data.Ey = new float[setup.size]();
    data.Ez = new float[setup.size]();
}

// Allocate the memory for medium on device.
void malloc_medium_d(const FDTD_Setup& setup, Medium_Params& medium)
{
    HIP_CHECK(hipMalloc(&medium.mu_r_x, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.mu_r_y, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.mu_r_z, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_m_x, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_m_y, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_m_z, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.epsilon_r_x, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.epsilon_r_y, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.epsilon_r_z, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_x, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_y, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&medium.sigma_z, setup.size * sizeof(float)));
}

// Allocate the memory for field components on device.
void malloc_field_d(const FDTD_Setup& setup, Field_Dims& data)
{
    HIP_CHECK(hipMalloc(&data.Hx, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&data.Hy, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&data.Hz, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&data.Ex, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&data.Ey, setup.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&data.Ez, setup.size * sizeof(float)));

    HIP_CHECK(hipMemset(data.Hx, 0, setup.size * sizeof(float)));
    HIP_CHECK(hipMemset(data.Hy, 0, setup.size * sizeof(float)));
    HIP_CHECK(hipMemset(data.Hz, 0, setup.size * sizeof(float)));
    HIP_CHECK(hipMemset(data.Ex, 0, setup.size * sizeof(float)));
    HIP_CHECK(hipMemset(data.Ey, 0, setup.size * sizeof(float)));
    HIP_CHECK(hipMemset(data.Ez, 0, setup.size * sizeof(float)));
}

// Copy the memory for medium.
void memcpy_medium(const FDTD_Setup&    setup,
                   Medium_Params&       dst,
                   const Medium_Params& src,
                   hipMemcpyKind        kind)
{
    HIP_CHECK(hipMemcpy(dst.mu_r_x, src.mu_r_x, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.mu_r_y, src.mu_r_y, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.mu_r_z, src.mu_r_z, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_m_x, src.sigma_m_x, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_m_y, src.sigma_m_y, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_m_z, src.sigma_m_z, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.epsilon_r_x, src.epsilon_r_x, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.epsilon_r_y, src.epsilon_r_y, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.epsilon_r_z, src.epsilon_r_z, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_x, src.sigma_x, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_y, src.sigma_y, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.sigma_z, src.sigma_z, setup.size * sizeof(float), kind));
}

// Copy the memory for field components.
void memcpy_field(const FDTD_Setup& setup,
                  Field_Dims&       dst,
                  const Field_Dims& src,
                  hipMemcpyKind     kind)
{
    HIP_CHECK(hipMemcpy(dst.Hx, src.Hx, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.Hy, src.Hy, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.Hz, src.Hz, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.Ex, src.Ex, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.Ey, src.Ey, setup.size * sizeof(float), kind));
    HIP_CHECK(hipMemcpy(dst.Ez, src.Ez, setup.size * sizeof(float), kind));
}

void write_component(const unsigned int nx,
                     const unsigned int ny,
                     const unsigned int nz,
                     const unsigned int size,
                     const float*       data,
                     const fs::path     path)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs << nx << " " << ny << " " << nz << "\n";

    ofs.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size * sizeof(float)));
    ofs.close();
}

void write_field(const FDTD_Setup& setup, const Field_Dims& data, const fs::path path)
{
    const unsigned int nx = setup.nx, ny = setup.ny, nz = setup.nz;
    write_component(nx, ny - 1, nz - 1, setup.size, data.Hx, path / "Hx.bin");
    write_component(nx - 1, ny, nz - 1, setup.size, data.Hy, path / "Hy.bin");
    write_component(nx - 1, ny - 1, nz, setup.size, data.Hz, path / "Hz.bin");
    write_component(nx - 1, ny, nz, setup.size, data.Ex, path / "Ex.bin");
    write_component(nx, ny - 1, nz, setup.size, data.Ey, path / "Ey.bin");
    write_component(nx, ny, nz - 1, setup.size, data.Ez, path / "Ez.bin");
}

// Free the memory for medium.
void cleanup_medium(Medium_Params& medium)
{
    delete[] medium.mu_r_x;
    delete[] medium.mu_r_y;
    delete[] medium.mu_r_z;
    delete[] medium.sigma_m_x;
    delete[] medium.sigma_m_y;
    delete[] medium.sigma_m_z;
    delete[] medium.epsilon_r_x;
    delete[] medium.epsilon_r_y;
    delete[] medium.epsilon_r_z;
    delete[] medium.sigma_x;
    delete[] medium.sigma_y;
    delete[] medium.sigma_z;
}

// Free the memory for field components.
void cleanup_field(Field_Dims& data)
{
    delete[] data.Hx;
    delete[] data.Hy;
    delete[] data.Hz;
    delete[] data.Ex;
    delete[] data.Ey;
    delete[] data.Ez;
}

// Free the memory for medium on device.
void cleanup_medium_d(Medium_Params& medium)
{
    HIP_CHECK(hipFree(medium.mu_r_x));
    HIP_CHECK(hipFree(medium.mu_r_y));
    HIP_CHECK(hipFree(medium.mu_r_z));
    HIP_CHECK(hipFree(medium.sigma_m_x));
    HIP_CHECK(hipFree(medium.sigma_m_y));
    HIP_CHECK(hipFree(medium.sigma_m_z));
    HIP_CHECK(hipFree(medium.epsilon_r_x));
    HIP_CHECK(hipFree(medium.epsilon_r_y));
    HIP_CHECK(hipFree(medium.epsilon_r_z));
    HIP_CHECK(hipFree(medium.sigma_x));
    HIP_CHECK(hipFree(medium.sigma_y));
    HIP_CHECK(hipFree(medium.sigma_z));
}

// Free the memory for field components on device.
void cleanup_field_d(Field_Dims& data)
{
    HIP_CHECK(hipFree(data.Hx));
    HIP_CHECK(hipFree(data.Hy));
    HIP_CHECK(hipFree(data.Hz));
    HIP_CHECK(hipFree(data.Ex));
    HIP_CHECK(hipFree(data.Ey));
    HIP_CHECK(hipFree(data.Ez));
}
