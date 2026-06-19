# PQC TrustFlow Frontend

Notebook-friendly `ipywidgets` frontend for the AMD ROCm PQC workflow.

## What this version does

- Scans a folder of demo documents.
- Builds a `manifest.json` with file names, sizes, and SHA-256 digests.
- Encrypts every file into a `.pqcpack.zip` package.
- Uses AES-256-GCM when `cryptography` is available in the Jupyter image.
- Restores and verifies the package.
- Creates a tampered copy and confirms detection.
- Calls the selected ROCm KEM and signature executables as proof runs and saves logs.

The current package authenticator is a demo-layer package MAC. The ROCm KEM/SIG
executables are still called and logged from the UI, and the next backend step is
to replace the package authenticator with direct ROCm file-I/O signing once the
minimal CLI mode is compiled into the HIP binaries.

## Quick start

Open `pqc_trustflow_widgets_demo.ipynb` in JupyterLab and run the second cell.

Or run this in a notebook from `/app/PQC_TrustFlow_ROCm`:

```python
from pqc_trustflow_frontend import launch_app
launch_app()
```

The default folder is `pqc_trustflow_frontend/sample_docs`. You can replace it
with any folder under `/app` that contains documents for the demo.

## Terminal smoke test

```bash
cd /app/PQC_TrustFlow_ROCm
python3 - <<'PY'
from pqc_trustflow_frontend.backends import ensure_sample_docs, create_secure_pack, create_tampered_copy_and_verify
src = ensure_sample_docs()
r = create_secure_pack(src, "Kyber-768", "ML-DSA-65", 128, "paper", run_rocm=True)
print("pack:", r.pack_dir)
print("verified:", r.verified)
print("rocm logs:", r.rocm_logs)
t = create_tampered_copy_and_verify(r.pack_dir)
print("tamper detected:", t["tamper_detected"])
PY
```
