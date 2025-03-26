# Applications FDTD Example

## Description

This example demonstrates how to implement a 3D [Finite-Difference Time-Domain (FDTD)](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) method on the GPU, incorporating Perfectly Matched Layers (PML) for absorbing boundary conditions. We use the Yee scheme to discretize Maxwell’s equations in both space and time, and then port the update steps to HIP kernels, and then compare the results to those from CPU implementation. For the reference of the implementation please check: [Understanding the FDTD Method](https://eecs.wsu.edu/~schneidj/ufdtd/)

### 1. Physical Context

In standard electromagnetics, **Maxwell’s equations** can be expressed in differential form as:

1. **Faraday’s Law (rotation of electric field)**

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}. $$

2. **Ampère-Maxwell Law (rotation of magnetic field)**

$$\nabla \times \mathbf{H} = \frac{\partial \mathbf{D}}{\partial t} + \mathbf{J},$$

where $\mathbf{E}$ is the electric field, $\mathbf{H}$ is the magnetic field, $\mathbf{B} = \mu \mathbf{H}$, and $\mathbf{D} = \varepsilon \mathbf{E}$. In a lossy medium (or PML region), we incorporate **electric conductivity** $\sigma$ and **magnetic loss** $\sigma_m$. These modify Ampère’s and Faraday’s laws by adding conduction currents $\sigma \mathbf{E}$ or magnetic loss terms $\sigma_m \mathbf{H}$.


### 2. From Continuous Equations to Discrete Updates

#### 2.1 Curl Form of Maxwell’s Equations

Under time-domain form and using the constitutive relationships $\mathbf{D} = \varepsilon \mathbf{E}$, $\mathbf{B} = \mu \mathbf{H}$, and including conductivities $(\sigma$ or $\sigma_m)$, we obtain:

- **Faraday’s Law**:

$$\nabla \times \mathbf{E} = -\frac{\partial (\mu \mathbf{H})}{\partial t} - \sigma_m   \mathbf{H},$$

(the $\sigma_m$ term represents a magnetic loss).

- **Ampère’s Law (with conduction)**:

$$\nabla \times \mathbf{H} = \frac{\partial (\varepsilon \mathbf{E})}{\partial t} + \sigma  \mathbf{E}.$$

#### 2.2 Yee Scheme Discretization

The Yee scheme staggers electric and magnetic field components both in space and in time. For instance, the component $E_x$ is stored on edges in the $x$-direction, while $H_y$ and $H_z$ might be offset by half a cell. Each time step alternates between updating:

1. The magnetic field $\mathbf{H}$ based on the curl of $\mathbf{E}$.
2. The electric field $\mathbf{E}$ based on the curl of $\mathbf{H}$.

In a lossy medium or PML region, the discrete update equations look like:

- **Magnetic Field Update** ($H_x$ as an example):

$$
H_x^{n+1} = \frac{\bigl(1 - \frac{\sigma_m \Delta t}{2 \mu}\bigr) H_x^n\  -\  \frac{\Delta t}{\mu} \mathrm{curl}_x(E)}{1 + \frac{\sigma_m \Delta t}{2 \mu}}.
$$

Here, $\mathrm{curl}_x(E) = \bigl(\partial E_z/\partial y - \partial E_y/\partial z\bigr)$ in discrete form.

- **Electric Field Update** ($E_x$ as an example):

$$
E_x^{n+1} = \frac{\bigl(1 - \frac{\sigma \Delta t}{2 \varepsilon}\bigr) E_x^n\  +\  \frac{\Delta t}{\varepsilon} \mathrm{curl}_x(H)}{1 + \frac{\sigma \Delta t}{2 \varepsilon}}.
$$

Here $\sigma$ and $\sigma_m$ are spatially dependent to represent different materials or PML layers.

### 3. Overview of the GPU Kernels

In this code, each update step is run on the GPU. We assign a thread to each $(x,y,z)$ point in the domain, and each kernel computes the relevant field component for that point:

1. **`apply_source_kernel`**
- Injects a time-harmonic or sinusoidal wave (the “source”) into the $\mathbf{E}$ field array.
- Demonstrates how to place an excitation at a specific $(x,y,z)$ without interfering with the rest of the domain.

2. **`updateHx_kernel`**
- Implements the discrete equation for $H_x$ (see formula above).
- Reads electric field data $(E_y, E_z)$ to compute the curl term, then applies the $\sigma_m$ factor to handle magnetic losses or PML.

3. **`updateEx_kernel`**
- Similar structure for $E_x$.
- Computes $\mathrm{curl}_x(H)$ from $(H_y, H_z)$, applies electric conductivity $\sigma$, and updates $E_x$.

Each kernel carefully accounts for **Yee grid staggering** (e.g., indexing offsets in x, y, z) and handles boundary conditions or PML by adjusting $\sigma, \sigma_m$.

### 4. Perfectly Matched Layers (PML) in Code

We add PML by gradually ramping up $\sigma$ and $\sigma_m$ near the domain edges. This ensures the wave is absorbed with minimal reflection. We define a function $\text{polynomial}\\_\text{sigma}$ that smoothly increases $\sigma$ toward the boundary. Similar logic applies to magnetic loss $\sigma_m$. The FDTD update kernels then incorporate these values, causing fields in the boundary region to decay rather than reflect back.


### 5. From Faraday’s Law to Discretized Curl

#### 5.1 Faraday’s Law to Magnetic Field Update

- **Continuous** (Faraday’s Law):

$$
\nabla \times \mathbf{E} = -\frac{\partial (\mu \mathbf{H})}{\partial t}.
$$

- **After rearranging**:

$$
\frac{\partial H_x}{\partial t} \ \propto\  \frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z}.
$$

- **Discrete** (Yee update):

$$
H_x^{n+1}(i,j,k) = H_x^n(i,j,k) - \frac{\Delta t}{\mu}\Bigl[\Delta_y(E_z)\  -\ \Delta_z(E_y)\Bigr] + \dots 
$$

(with additional factors for $\sigma_m$)

#### 5.2 Ampère’s Law to Electric Field Update

- **Continuous** (Ampère-Maxwell + conduction):

$$
\nabla \times \mathbf{H} = \frac{\partial (\varepsilon \mathbf{E})}{\partial t} + \sigma \mathbf{E}. 
$$

- **After rearranging**:

$$
\frac{\partial E_x}{\partial t} \ \propto\  \frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z} - \frac{\sigma}{\varepsilon} E_x. 
$$

- **Discrete**:

$$
E_x^{n+1}(i,j,k) = \frac{\bigl(1 - \tfrac{\sigma \Delta t}{2 \varepsilon}\bigr) E_x^n + \tfrac{\Delta t}{\varepsilon}\Bigl[\Delta_y(H_z) -\Delta_z(H_y)\Bigr]}{1 + \tfrac{\sigma \Delta t}{2 \varepsilon}}. 
$$

Here, conduction or PML damping enters through $\sigma$, while magnetic losses enter via $\sigma_m$ in the $H$-field updates.

**Note**: This example omits full boundary checks, advanced PML tuning (e.g., $\kappa$, $\alpha$), and memory optimizations. Readers should refer to specialized FDTD textbooks or the original Berenger and S. D. Gedney CPML references for more rigorous derivations. Nonetheless, the essential workflow for GPU-accelerated FDTD is illustrated:

1. Represent Maxwell’s Equations in curl form.
2. Discretize with the Yee scheme.
3. Incorporate conduction or PML damping.
4. Execute updates.

### Application flow

1. Parse user input.
2. Allocate and initialize memories for computation on CPU and GPU.
3. Initialize the emulation setup and medium parameters.
4. Enter the wave propagation loop, first the CPU implementation, then parallize the GPU kernels and run.
5. Report the CPU and GPU computation time.
5. Check the consistency between CPU and GPU results.
6. If specified, output the results to binary files.

### Command line interface
- `-t <timestep>` sets `timestep` as the total number of timesteps for the emulation iteration.
- `-s <sample>` sets `sample` as the number of timesteps between two consistency check and result output.
- `-o <output>` sets `output` as the target directory for result output.

## Key APIs and Concepts

1. **Device Memory Management**
- **`hipMalloc` / `hipFree`**: Used for allocating and freeing device memory。
- **`hipMemcpy`**: Copies data between host and device (e.g., `hipMemcpyHostToDevice` and `hipMemcpyDeviceToHost`).

2. **Kernel Launch**
- **`kernelName<<<grid, block, 0, stream>>>`**: Enqueues a kernel call on the GPU.

3. **Streams and Synchronization**
- **`hipStreamCreate`**: Creates a stream for kernel launches.
- **`hipStreamSynchronize`**: Waits until all commands in a given stream have completed.

4. **Error Checking**
- **`HIP_CHECK(...)`** is a macro that wraps HIP calls (such as `hipMalloc`) and reports any error returned by them.

## Demonstrated API Calls

### HIP runtime

#### Device symbols
- `blockDim`
- `blockIdx`
- `threadIdx`

#### Host symbols
- `__global__`
- `__host__`
- `__device__`
- `__forceinline__`
- `hipMalloc`
- `hipFree`
- `hipMemcpy`
- `hipMemcpyHostToDevice`
- `hipMemcpyDeviceToHost`
- `hipStreamCreate`
- `hipStreamSynchronize`
