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

# Default configurations for automated runs (push/PR)
INSTALL_METHODS = ["wheel", "tarball"]

# Distros to build against – keyed by short name.
# Add new entries here to enable more distros (also add to workflow_dispatch options).
DISTRO_MAP = {
    "ubuntu-22.04": {"image": "ghcr.io/rocm/rocm-examples-ubuntu-22.04:latest", "label": "Ubuntu 22.04"},
    "sles-15.7":    {"image": "ghcr.io/rocm/rocm-examples-sles-15.7:latest",    "label": "SLES 15.7"},
    "rhel-8":       {"image": "ghcr.io/rocm/rocm-examples-rhel-8:latest",       "label": "RHEL 8"},
}

def _is_all(value):
    """Return True when the input means 'use everything'."""
    return not value or value == "all"

def main():
    gpu_input = os.getenv("GPU_CONFIG", "")
    install_input = os.getenv("INSTALL_METHOD", "")
    distro_input = os.getenv("DISTRO", "")

    # Determine GPU configurations
    if _is_all(gpu_input):
        gpu_targets = list(GPU_CONFIG_MAP.keys())
    else:
        if gpu_input not in GPU_CONFIG_MAP:
            raise ValueError(f"Invalid GPU target: {gpu_input}. Allowed: {list(GPU_CONFIG_MAP.keys())}")
        gpu_targets = [gpu_input]

    # Determine install methods
    if _is_all(install_input):
        install_methods = INSTALL_METHODS
    else:
        if install_input not in INSTALL_METHODS:
            raise ValueError(f"Invalid install method: {install_input}. Allowed: {INSTALL_METHODS}")
        install_methods = [install_input]

    # Determine distros
    if _is_all(distro_input):
        distro_keys = list(DISTRO_MAP.keys())
    else:
        if distro_input not in DISTRO_MAP:
            raise ValueError(f"Invalid distro: {distro_input}. Allowed: {list(DISTRO_MAP.keys())}")
        distro_keys = [distro_input]

    gpu_configs = [
        {"gpu_target": t, "therock_family": GPU_CONFIG_MAP.get(t, "gfx110X-all")}
        for t in gpu_targets
    ]

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"gpu_configs={json.dumps(gpu_configs)}\n")
            f.write(f"install_methods={json.dumps(install_methods)}\n")
            f.write(f"distros={json.dumps(distro_keys)}\n")
            f.write(f"distro_map={json.dumps(DISTRO_MAP)}\n")

    print(f"gpu_configs={json.dumps(gpu_configs)}")
    print(f"install_methods={json.dumps(install_methods)}")
    print(f"distros={json.dumps(distro_keys)}")
    print(f"distro_map={json.dumps(DISTRO_MAP)}")

if __name__ == "__main__":
    main()
