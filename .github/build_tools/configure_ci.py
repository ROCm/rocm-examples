#!/usr/bin/env python3
"""Configure CI matrix based on workflow inputs or defaults."""
import os
import json

# GPU target to TheRock family mapping.
# Only include targets that have self-hosted runners available.
# To add a target: uncomment its entry AND ensure a runner with that label exists.
GPU_CONFIG_MAP = {
    "gfx1100": "gfx110X-all",
    "gfx1151": "gfx1151",
    # "gfx1201": "gfx120X-all",  # no runner yet
    # "gfx90a":  "gfx90X-dcgpu", # no runner yet
    # "gfx942":  "gfx94X-dcgpu", # no runner yet
}

# Install methods for all distros (ROCm installed at CI runtime from TheRock nightlies).
INSTALL_METHODS = ["whl-multi-arch", "tarball-multi-arch"]

# Distros to build against – keyed by short name.
# "install_methods": omit to use the global INSTALL_METHODS list.
# Add new entries here to enable more distros (also add to workflow_dispatch options).
DISTRO_MAP = {
    # Multi-arch images: ROCm installed at CI runtime from TheRock nightlies
    "sles-15.7":    {"image": "ghcr.io/rocm/rocm-examples-sles-15.7-multiarch:latest",    "label": "SLES 15.7"},
    "almalinux-8":  {"image": "ghcr.io/rocm/rocm-examples-almalinux-8-multiarch:latest",  "label": "AlmaLinux 8"},
    "ubuntu-24.04": {"image": "ghcr.io/rocm/rocm-examples-ubuntu-24.04-multiarch:latest", "label": "Ubuntu 24.04"},
    "ubuntu-26.04": {"image": "ghcr.io/rocm/rocm-examples-ubuntu-26.04-multiarch:latest", "label": "Ubuntu 26.04"},
    # Disabled until CI validation is complete — Dockerfiles in Scripts/MultiArch/
    # "rocky-9":   {"image": "ghcr.io/rocm/rocm-examples-rocky-9-multiarch:latest",    "label": "Rocky Linux 9",   "install_methods": ["whl-multi-arch"]},
    # "rhel-10.1": {"image": "ghcr.io/rocm/rocm-examples-rhel-10.1-multiarch:latest",  "label": "RHEL 10.1",       "install_methods": ["whl-multi-arch"]},
    # "oracle-10": {"image": "ghcr.io/rocm/rocm-examples-oracle-10-multiarch:latest",  "label": "Oracle Linux 10", "install_methods": ["whl-multi-arch"]},
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

    # Build distro_map output: resolve per-distro install_methods, strip the key from output.
    distro_map_out = {}
    for key in distro_keys:
        entry = DISTRO_MAP[key]
        distro_map_out[key] = {
            "image": entry["image"],
            "label": entry["label"],
            "install_methods": entry.get("install_methods", install_methods),
        }

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"gpu_configs={json.dumps(gpu_configs)}\n")
            f.write(f"install_methods={json.dumps(install_methods)}\n")
            f.write(f"distros={json.dumps(distro_keys)}\n")
            f.write(f"distro_map={json.dumps(distro_map_out)}\n")

    print(f"gpu_configs={json.dumps(gpu_configs)}")
    print(f"install_methods={json.dumps(install_methods)}")
    print(f"distros={json.dumps(distro_keys)}")
    print(f"distro_map={json.dumps(distro_map_out)}")

if __name__ == "__main__":
    main()
