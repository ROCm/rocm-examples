"""Write GITHUB_ENV, GITHUB_OUTPUT, and GITHUB_STEP_SUMMARY for a whl-multi-arch install."""

import os
import subprocess
import sysconfig

venv = os.environ.get("VIRTUAL_ENV", "/opt/rocm-venv")
site = sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv})
rocm = site + "/_rocm_sdk_devel"
core = site + "/_rocm_sdk_core/lib"
libs = site + "/_rocm_sdk_libraries/lib"

with open(os.environ["GITHUB_ENV"], "a") as f:
    f.write(f"ROCM_PATH={rocm}\n")
    f.write(f"HIP_PLATFORM=amd\n")
    f.write(f"HIP_PATH={rocm}\n")
    f.write(f"HIP_DEVICE_LIB_PATH={rocm}/lib/llvm/amdgcn/bitcode\n")
    f.write(f"PATH={rocm}/bin:{rocm}/llvm/bin:{venv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n")
    f.write(f"CPATH={rocm}/include\n")
    f.write(f"PKG_CONFIG_PATH={rocm}/lib/pkgconfig\n")
    f.write(f"LIBRARY_PATH={rocm}/lib:{rocm}/lib64\n")
    f.write(f"LD_LIBRARY_PATH={core}:{libs}:{rocm}/lib:{rocm}/llvm/lib\n")
    # whl-multi-arch omits amdllvm needed for OpenMP GPU offloading
    f.write("ENABLE_OPENMP=OFF\n")
    # hipDNN headers require C++20; system g++ on AlmaLinux 8 is too old
    f.write(f"CXX={venv}/bin/amdclang++\n")

rocm_version = subprocess.check_output(["rocm-sdk", "version"], text=True).strip()

with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"rocm_version={rocm_version}\n")

with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
    f.write(f"## ROCm Version: {rocm_version} (whl-multi-arch)\n")
