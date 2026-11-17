#!/bin/bash
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

EXAMPLE_QUERY="rocprofv3-avail"
EXAMPLE_TOOL="rocprofv3"
EXAMPLE_BIN="${EXAMPLE_TOOL}-matmul"
EXAMPLE_BIN_ROCTX="${EXAMPLE_BIN}-roctx"
EXAMPLE_WORKLOAD="./$EXAMPLE_BIN"
EXAMPLE_WORKLOAD_ROCTX="./${EXAMPLE_BIN_ROCTX}"

# Check for existence of tools
REQUIRED_TOOLS="$EXAMPLE_QUERY $EXAMPLE_TOOL"
MISSING_TOOLS=""

for tool in $REQUIRED_TOOLS; do
    if ! [ -x "$(command -v $tool)" ]; then
        MISSING_TOOLS="$MISSING_TOOLS $tool"
    fi
done

if [ -n "$MISSING_TOOLS" ]; then
    echo "Error: Could not find the following tools in PATH:$MISSING_TOOLS" >&2
    exit 1
fi

if [[ ! -f "$EXAMPLE_BIN" || ! -f "$EXAMPLE_BIN_ROCTX" ]]; then
    echo "Error: $EXAMPLE_BIN or $EXAMPLE_BIN_ROCTX not present in working directory" >&2
    exit 1
fi

# Exit on any error
set -e

echo "==============================================================================="
echo "Kernel name filtering"
echo "==============================================================================="
# Kernels are specified as regular expressions in a YAML file. Note that YAML files can contain more than just filters;
# for the sake of easy to follow examples they are split into multiple files here.
OUTDIR="${EXAMPLE_TOOL}-kernel-filter" \
    $EXAMPLE_TOOL \
    --runtime-trace \
    --input kernel_filter.yml \
    --output-directory %env{OUTDIR}% \
    --output-format pftrace \
    -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Collecting PMC counters"
echo "==============================================================================="
# PMC counters are specified in a YAML file. A device's available PMC counters can be obtained by calling:
# rocprofv3-avail info --pmc
OUTDIR="${EXAMPLE_TOOL}-pmc-counters" \
    $EXAMPLE_TOOL \
    --input wavefront_stats.yml \
    --output-directory %env{OUTDIR}% \
    --output-format pftrace \
    -- $EXAMPLE_WORKLOAD

echo "==============================================================================="
echo "Instrumenting with rocTX"
echo "==============================================================================="
# The following performs a trace of an application with user-defined rocTX instrumentation.
OUTDIR="${EXAMPLE_TOOL}-rocTX" \
    $EXAMPLE_TOOL \
    --marker-trace \
    --output-directory %env{OUTDIR}% \
    --output-format pftrace \
    -- $EXAMPLE_WORKLOAD_ROCTX

echo "==============================================================================="
echo "PC sampling"
echo "==============================================================================="
# PC sampling is currently a beta feature and not supported on all devices. The supported method, unit and interval
# range vary per agent and ROCm version, so query the capabilities of the device the workload actually runs on
# (HIP's default device, ordinal 0, since the workload never calls hipSetDevice) and use its first reported
# configuration instead of hardcoding one or picking up a config reported for a different agent.
#
# rocprofv3-avail's "GPU:<id>" is its own profiler agent index, not HIP's device ordinal. HIP_VISIBLE_DEVICES
# (falling back to CUDA_VISIBLE_DEVICES, matching HIP's own fallback) reassigns ordinals by the literal order of
# its comma-separated list -- e.g. "3,2" makes physical GPU 3 ordinal 0 -- a HIP-runtime-private remap that
# rocprofv3-avail's listing never reflects, so it can't be recovered by inspecting that listing; only the literal
# env var text gives the order. ROCR_VISIBLE_DEVICES needs no such handling: it filters the agent enumeration
# below the HIP layer, so rocprofv3-avail already observes the same reordering HIP does. When neither
# HIP_VISIBLE_DEVICES nor CUDA_VISIBLE_DEVICES is set, no HIP-private remap exists and GPU:<id> already matches
# HIP's ordinals directly.
HIP_VISIBLE_LIST="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
if [[ -z "$HIP_VISIBLE_LIST" ]]; then
    HIP_DEVICE0_GPU=0
else
    HIP_DEVICE0_GPU="${HIP_VISIBLE_LIST%%,*}"
    # A UUID (or anything else that isn't a plain index) can't be resolved against rocprofv3-avail's listing,
    # which has no UUID field to match against.
    [[ "$HIP_DEVICE0_GPU" =~ ^[0-9]+$ ]] || HIP_DEVICE0_GPU=""
fi

if [[ -z "$HIP_DEVICE0_GPU" ]]; then
    echo "PC sampling not supported: could not resolve HIP's default device to a profiler agent"
else
    # rocprofv3-avail's `-d` does not scope `info --pc-sampling` to a single agent: it always lists every
    # PC-sampling-capable agent. The awk parser below therefore filters the listing to the resolved GPU's block
    # itself, so a config is never picked up from a different agent.
    PC_SAMPLING_INFO="$($EXAMPLE_QUERY info --pc-sampling)"
    PC_SAMPLING_CONFIG="$(awk -v target="$HIP_DEVICE0_GPU" '
        # Reset on every GPU boundary, and only keep fields while inside the matching GPU block, so a config is
        # never built out of fields left over from, or belonging to, a different agent'"'"'s section.
        /^[[:space:]]*GPU[[:space:]]*:/           { selected = ($NF == target); method = ""; unit = ""; min = ""; next }
        !selected                                 { next }
        /^[[:space:]]*Method[[:space:]]*:/        { method = $NF; next }
        /^[[:space:]]*Unit[[:space:]]*:/          { unit = $NF; next }
        /^[[:space:]]*Min_Interval[[:space:]]*:/  { min = $NF; next }
        /^[[:space:]]*Max_Interval[[:space:]]*:/  { if (method != "") { print method, unit, min; exit } }
    ' <<< "$PC_SAMPLING_INFO")"

    if [[ -z "$PC_SAMPLING_CONFIG" ]]; then
        echo "PC sampling not supported on device 0 (profiler agent GPU:$HIP_DEVICE0_GPU)"
    else
        read -r METHOD UNIT INTERVAL <<< "$PC_SAMPLING_CONFIG"

        # --pc-sampling-unit only accepts the plural choices (instructions/cycles/time); the capability listing above
        # reports them singular except for "time".
        case "$UNIT" in
            instruction*) UNIT=instructions ;;
            cycle*)       UNIT=cycles ;;
        esac

        OUTDIR="${EXAMPLE_TOOL}-pc-sampling" \
            $EXAMPLE_TOOL \
            --pc-sampling-beta-enabled \
            --pc-sampling-unit "$UNIT" \
            --pc-sampling-method "$METHOD" \
            --pc-sampling-interval "$INTERVAL" \
            --output-directory %env{OUTDIR}% \
            --output-format csv \
            -- $EXAMPLE_WORKLOAD
    fi
fi

exit 0
