# PQC TrustFlow ROCm 快速运行说明

本文档用于评审或复现实验时快速启动前端、执行完整流程，并确认输出结果是否正确。

## 1. 进入项目目录

在 AMD JupyterLab 服务器终端中执行：

```bash
cd /app/PQC_TrustFlow_ROCm
export LD_LIBRARY_PATH=/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:$LD_LIBRARY_PATH
```

如果项目目录尚未改名，也可以先在当前解压目录中运行；最终提交版本建议统一使用 `/app/PQC_TrustFlow_ROCm`。

## 2. 启动 Notebook 前端

打开 `pqc_trustflow_widgets_demo.ipynb`，执行：

```python
%cd /app/PQC_TrustFlow_ROCm

import os
os.environ["LD_LIBRARY_PATH"] = "/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from pqc_trustflow_frontend import launch_app
launch_app()
```

启动后会显示 PQC TrustFlow 前端界面。推荐配置：

```text
KEM: Kyber-768
SIG: ML-DSA-65
batch: 128
mode: paper
```

## 3. 前端按钮含义

`准备`：生成或检查演示输入文件，并初始化流程状态。

`生成安全包`：调用 ROCm KEM 文件级接口生成共享密钥材料，使用 AES-256-GCM 加密文件，再调用 ROCm ML-DSA/Aigis-sig 文件级接口签名 manifest。

`查看安全包`：显示本次生成的安全包目录、zip 包、密文文件、KEM ciphertext 和 manifest 信息。

`查看证明`：显示签名载荷、签名文件、ROCm 后端日志、KEM/SIG API 调用结果。

`解包并验证`：执行 KEM decaps，恢复 AES 密钥，解密文件，校验 SHA-256 摘要，并验证 manifest 签名。

`篡改测试`：复制安全包，自动篡改一个密文或摘要相关文件，再重新验证，确认系统能检测异常。

`查看恢复目录`：查看解密后恢复出的文件。

`一键运行`：自动执行准备、生成安全包、解包并验证，适合快速演示。

`重置`：清空当前前端状态，重新开始一次流程。

## 4. 期望前端结果

正常流程中，`流程` 标签页应显示：

```text
准备: PASS
生成安全包: PASS
解包验证: PASS
```

`结果与证据` 标签页应包含：

```text
正常包验证: PASS
KEM 后端: ROCm KEM batch file API
签名后端: ROCm ML-DSA/Aigis-sig batch file API
KEM ciphertext: kem/kem_ct.bin
签名载荷: sig/manifest.payload.json
签名文件: sig/manifest.sig
```

执行 `篡改测试` 后，期望结果为：

```text
篡改检测: PASS
篡改包验证结果: FAIL
```

这表示正常包可以通过解密、验签和摘要校验；被篡改后的包无法通过验证。

## 5. 终端一键验证

如果需要在终端生成一份可归档的 smoke test 输出，可执行：

```bash
cd /app/PQC_TrustFlow_ROCm
mkdir -p results/smoke_tests results/logs results/screenshots
export LD_LIBRARY_PATH=/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:$LD_LIBRARY_PATH

python3 - <<'PY' | tee results/smoke_tests/trustflow_smoke_$(date +%Y%m%d_%H%M%S).txt
from pqc_trustflow_frontend.backends import ensure_sample_docs, create_secure_pack, create_tampered_copy_and_verify
import json
from pathlib import Path

src = ensure_sample_docs()
r = create_secure_pack(src, "Kyber-768", "ML-DSA-65", 128, "paper", run_rocm=True)

print("pack:", r.pack_dir)
print("zip:", r.pack_zip)
print("verified:", r.verified)
print("logs:", json.dumps(r.rocm_logs, ensure_ascii=False, indent=2))
print("notes:", r.notes)

m = json.loads(Path(r.manifest_path).read_text())
print("kem_backend:", m.get("kem_backend"))
print("signature_backend:", m.get("signature_backend"))
print("kem_ciphertext_file:", m.get("kem_ciphertext_file"))
print("sig_payload:", m.get("sig_payload"))
print("manifest_signature:", m.get("manifest_signature"))

t = create_tampered_copy_and_verify(r.pack_dir)
print("tamper_detected:", t["tamper_detected"])
print("tamper_verified:", t["verified"])
print("file_errors:", t["file_errors"])
print("kem_ok:", t.get("kem_ok"))
print("sig_api_ok:", t.get("sig_api_ok"))
PY
```

期望关键输出：

```text
verified: True
notes: []
kem_backend: ROCm KEM batch file API
signature_backend: ROCm ML-DSA/Aigis-sig batch file API
tamper_detected: True
tamper_verified: False
```

## 6. 结果文件归档建议

运行完成后，建议保留以下证据：

```text
results/
  screenshots/
    01_frontend_full_ui.png
    02_pack_encrypt_sign.png
    03_decrypt_verify_digest.png
    04_tamper_detection.png
    05_repository_layout.png
    06_one_click_test.png
    07_generated_artifacts.png
  smoke_tests/
    trustflow_smoke_*.txt
  logs/
    kemapi_keygen_sample.log
    kemapi_encaps_sample.log
    kemapi_decaps_sample.log
    sigapi_sign_sample.log
    sigapi_verify_sample.log
```

其中 `screenshots/` 放人工截图，`smoke_tests/` 放终端一键测试输出，`logs/` 放关键 ROCm KEM/SIG 文件级 API 日志样例。
