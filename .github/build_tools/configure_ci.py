#!/usr/bin/env python3
"""Configure CI matrix based on workflow inputs or defaults."""
import os
import json

# GPU target to TheRock family mapping
GPU_CONFIG_MAP = {
    "gfx1100": "gfx110X-all",
    "gfx90a": "gfx90X-dcgpu",
    "gfx942": "gfx94X-dcgpu",
}

# Default configurations for automated runs (push/PR)
DEFAULT_GPU_TARGETS = ["gfx1100"]
DEFAULT_INSTALL_METHODS = ["wheel", "tarball"]


def main():
    # Read inputs from environment (set by workflow)
    gpu_input = os.getenv("GPU_CONFIG", "")
    install_input = os.getenv("INSTALL_METHOD", "")

    # Determine GPU configurations
    if gpu_input:
        # Manual dispatch: use the single provided value
        gpu_targets = [gpu_input]
    else:
        # Automated run: use all defaults
        gpu_targets = DEFAULT_GPU_TARGETS

    # Determine install methods
    if install_input:
        # Manual dispatch: use the single provided value
        install_methods = [install_input]
    else:
        # Automated run: use all defaults
        install_methods = DEFAULT_INSTALL_METHODS

    # Build gpu_config array with both gpu_target and therock_family
    gpu_configs = []
    for target in gpu_targets:
        family = GPU_CONFIG_MAP.get(target, "gfx110X-all")
        gpu_configs.append({
            "gpu_target": target,
            "therock_family": family
        })

    # Write outputs to $GITHUB_OUTPUT
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"gpu_configs={json.dumps(gpu_configs)}\n")
            f.write(f"install_methods={json.dumps(install_methods)}\n")
        print(f"Wrote outputs to {github_output}")
    else:
        # Local testing
        print(f"gpu_configs={json.dumps(gpu_configs)}")
        print(f"install_methods={json.dumps(install_methods)}")


if __name__ == "__main__":
    main()  
