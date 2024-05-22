# Using Address Sanitizer with a Short HIP Application

Consider the following simple and short demo of using the Address Sanitizer with a HIP application:

```C++
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 1;
}

int
main(int argc, char **argv)
{
    int m = std::atoi(argv[1]);
    int n1 = std::atoi(argv[2]);
    int n2 = std::atoi(argv[3]);
    int c = std::atoi(argv[4]);
    int *dp;
    hipMalloc(&dp, m*sizeof(int));
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
    int *hp = (int*)malloc(c * sizeof(int));
    hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    hipFree(dp);
    free(hp);
    std::puts("Done.");
    return 0;
}
```

This application will attempt to access invalid addresses for certain command line arguments. In particular, if `m < n1 * n2` some device threads will attempt to access
unallocated device memory.
Or, if `c < m`, the `hipMemcpy` function will copy past the end of the `malloc` allocated memory.
**Note**: The `hipcc` compiler is used here for simplicity. Compiling without XNACK results in a warning.
> `$ hipcc -g --offload-arch=gfx90a:xnack- -fsanitize=address
-shared-libsan mini.hip -o mini`
> `clang++: warning: ignoring '-fsanitize=address' option for offload arch 'gfx90a:xnack-'
as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead [-Woption-ignored]`
The binary compiled above will run, but the GPU code will not be
instrumented and the `m < n1 * n2` error will not be detected.
Switching to `--offload-arch=gfx90a:xnack+` in the command above
results in a warning-free compilation and an instrumented application.
After setting `PATH`, `LD_LIBRARY_PATH` and `HSA_XNACK` as described
earlier, a check of the binary with `ldd` yields

```shell
$ ldd mini
        linux-vdso.so.1 (0x00007ffd1a5ae000)
        libclang_rt.asan-x86_64.so => /opt/rocm-5.7.0-99999/llvm/lib/clang/17.0.0/lib/linux/libclang_rt.asan-x86_64.so (0x00007fb9c14b6000)
        libamdhip64.so.5 => /opt/rocm-5.7.0-99999/lib/asan/libamdhip64.so.5 (0x00007fb9bedd3000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb9beba8000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fb9bea59000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fb9bea3e000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fb9be84a000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fb9be844000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fb9be821000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fb9be817000)
        libamd_comgr.so.2 => /opt/rocm-5.7.0-99999/lib/asan/libamd_comgr.so.2 (0x00007fb9b4382000)
        libhsa-runtime64.so.1 => /opt/rocm-5.7.0-99999/lib/asan/libhsa-runtime64.so.1 (0x00007fb9b3b00000)
        libnuma.so.1 => /lib/x86_64-linux-gnu/libnuma.so.1 (0x00007fb9b3af3000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fb9c2027000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fb9b3ad7000)
        libtinfo.so.6 => /lib/x86_64-linux-gnu/libtinfo.so.6 (0x00007fb9b3aa7000)
        libelf.so.1 => /lib/x86_64-linux-gnu/libelf.so.1 (0x00007fb9b3a89000)
        libdrm.so.2 => /opt/amdgpu/lib/x86_64-linux-gnu/libdrm.so.2 (0x00007fb9b3a70000)
        libdrm_amdgpu.so.1 => /opt/amdgpu/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1 (0x00007fb9b3a62000)
```

This confirms that the address sanitizer runtime is linked in, and the ASAN instrumented version of the runtime libraries are used.
Checking the `PATH` yields

```shell
$ which llvm-symbolizer
/opt/rocm-5.7.0-99999/llvm/bin/llvm-symbolizer
```

Lastly, a check of the OS kernel version yields

```shell
$ uname -rv
5.15.0-73-generic #80~20.04.1-Ubuntu SMP Wed May 17 14:58:14 UTC 2023
```

which indicates that the required HMM support (kernel version > 5.6) is available.
This completes the necessary setup.
Running with `m = 100`, `n1 = 11`, `n2 = 10` and `c = 100` should produce
a report for an invalid access by the last 10 threads.

```text
=================================================================
==3141==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc 0x7fb1410d2cc4
WRITE of size 4 in workgroup id (10,0,0)
  #0 0x7fb1410d2cc4 in set1(int*) at /home/dave/mini/mini.cpp:0:10

Thread ids and accessed addresses:
00 : 0x7fb14371d190 01 : 0x7fb14371d194 02 : 0x7fb14371d198 03 : 0x7fb14371d19c 04 : 0x7fb14371d1a0 05 : 0x7fb14371d1a4 06 : 0x7fb14371d1a8 07 : 0x7fb14371d1ac
08 : 0x7fb14371d1b0 09 : 0x7fb14371d1b4

0x7fb14371d190 is located 0 bytes after 400-byte region [0x7fb14371d000,0x7fb14371d190)
allocated by thread T0 here:
    #0 0x7fb151c76828 in hsa_amd_memory_pool_allocate /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors.cpp:692:3
    #1 ...

    #12 0x7fb14fb99ec4 in hipMalloc /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:568:3
    #13 0x226630 in hipError_t hipMalloc<int>(int**, unsigned long) /opt/rocm-5.7.0-99999/include/hip/hip_runtime_api.h:8367:12
    #14 0x226630 in main /home/dave/mini/mini.cpp:19:5
    #15 0x7fb14ef02082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16

Shadow bytes around the buggy address:
  0x7fb14371cf00: ...

=>0x7fb14371d180: 00 00[fa]fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x7fb14371d200: ...

Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  ...
==3141==ABORTING
```

Running with `m = 100`, `n1 = 10`, `n2 = 10` and `c = 99` should produce a report for an invalid copy.

```text
=================================================================
==2817==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x514000150dcc at pc 0x7f5509551aca bp 0x7ffc90a7ae50 sp 0x7ffc90a7a610
WRITE of size 400 at 0x514000150dcc thread T0
    #0 0x7f5509551ac9 in __asan_memcpy /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors_memintrinsics.cpp:61:3
    #1 ...

    #9 0x7f5507462a28 in hipMemcpy_common(void*, void const*, unsigned long, hipMemcpyKind, ihipStream_t*) /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:637:10
    #10 0x7f5507464205 in hipMemcpy /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:642:3
    #11 0x226844 in main /home/dave/mini/mini.cpp:22:5
    #12 0x7f55067c3082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16
    #13 0x22605d in _start (/home/dave/mini/mini+0x22605d)

0x514000150dcc is located 0 bytes after 396-byte region [0x514000150c40,0x514000150dcc)
allocated by thread T0 here:
    #0 0x7f5509553dcf in malloc /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_malloc_linux.cpp:69:3
    #1 0x226817 in main /home/dave/mini/mini.cpp:21:21
    #2 0x7f55067c3082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16

SUMMARY: AddressSanitizer: heap-buffer-overflow /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors_memintrinsics.cpp:61:3 in __asan_memcpy
Shadow bytes around the buggy address:
  0x514000150b00: ...

=>0x514000150d80: 00 00 00 00 00 00 00 00 00[04]fa fa fa fa fa fa
  0x514000150e00: ...

Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  ...
==2817==ABORTING
```
