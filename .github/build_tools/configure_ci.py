#!/usr/bin/env python3
"""Configure CI matrix based on workflow inputs or defaults."""
import os
import json

# GPU target to TheRock family mapping
GPU_CONFIG_MAP = {
    "gfx1100": "gfx110X-all",
    "gfx1151": "gfx1151",
    # "gfx1201": "gfx120X-all",
    # "gfx90a": "gfx90X-dcgpu",
    # "gfx942": "gfx94X-dcgpu",
}

# Linux distribution to base image mapping
DISTRO_IMAGE_MAP = {
    "ubuntu-22.04": "ghcr.io/rocm/rocm-examples-ubuntu-22.04:latest",
    "sles-15.7": "ghcr.io/rocm/rocm-examples-sles-15.7:latest",
}

# Default configurations for automated runs (push/PR)
INSTALL_METHODS = ["wheel", "tarball"]

def main():
    # Read inputs from environment (set by workflow)
    gpu_input = os.getenv("GPU_CONFIG", "")
    install_input = os.getenv("INSTALL_METHOD", "")
    distro_input = os.getenv("DISTRO", "")


    # Determine GPU configurations
    if gpu_input:
        if gpu_input not in GPU_CONFIG_MAP:
            raise ValueError(f"Invalid GPU target: {gpu_input}. Allowed: {list(GPU_CONFIG_MAP.keys())}")
        gpu_targets = [gpu_input]
    else:
        # Automated run: use all allowed targets
        gpu_targets = list(GPU_CONFIG_MAP.keys())

    # Determine install methods
    if install_input:
        if install_input not in INSTALL_METHODS:
            raise ValueError(f"Invalid install method: {install_input}. Allowed: {INSTALL_METHODS}")
        install_methods = [install_input]
    else:
        # Automated run: use all allowed methods
        install_methods = INSTALL_METHODS

    if distro_input:
        if distro_input not in DISTRO_IMAGE_MAP:
            raise ValueError(f"Invalid Linux distribution: {distro_input}. Allowed: {list(DISTRO_IMAGE_MAP.keys())}")
        distros = [DISTRO_IMAGE_MAP[distro_input]]
    else:
        distros = list(DISTRO_IMAGE_MAP.values())

    gpu_configs = []
    for target in gpu_targets:
        family = GPU_CONFIG_MAP.get(target, "gfx110X-all")
        gpu_configs.append({
            "gpu_target": target,
            "therock_family": family
        })

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"gpu_configs={json.dumps(gpu_configs)}\n")
            f.write(f"install_methods={json.dumps(install_methods)}\n")
            f.write(f"distros={json.dumps(distros)}\n")

    print(f"gpu_configs={json.dumps(gpu_configs)}")
    print(f"install_methods={json.dumps(install_methods)}")
    print(f"distros={json.dumps(distros)}")
    
if __name__ == "__main__":
    main()  
