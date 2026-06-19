# 文件级 KEM/SIG 接口说明

本文档说明 PQC TrustFlow ROCm 前端实际调用的后端文件级接口。项目目标不是只展示单项 benchmark，而是提供可被前端和应用流程调用的后量子密码组件。

## 1. 总体数据流

发送端：

```text
输入文件
  -> 计算 SHA-256 摘要
  -> Kyber/Aigis-enc KEM encaps 得到 shared secret 和 KEM ciphertext
  -> SHA-256(shared secret) 派生 AES-256-GCM key
  -> AES-256-GCM 加密每个文件
  -> 生成 manifest
  -> ML-DSA/Aigis-sig 对 manifest payload 签名
  -> 输出 pqcpack 安全包
```

接收端：

```text
pqcpack 安全包
  -> Kyber/Aigis-enc KEM decaps 恢复 shared secret
  -> SHA-256(shared secret) 派生 AES-256-GCM key
  -> 验证 manifest 签名
  -> AES-256-GCM 解密文件
  -> 校验 SHA-256 摘要
  -> 输出恢复目录
```

## 2. KEM 文件级接口

可执行文件：

```text
kyberandaigis-enc/kyber768_amd
```

密钥生成：

```bash
./kyber768_amd \
  --api-kem-keygen \
  --batch 128 \
  --pk-out kem_pk.bin \
  --sk-out receiver_sk.demo_secret
```

封装：

```bash
./kyber768_amd \
  --api-kem-encaps \
  --batch 128 \
  --pk-in kem_pk.bin \
  --ct-out kem_ct.bin \
  --ss-out ss_sender.demo_secret
```

解封装：

```bash
./kyber768_amd \
  --api-kem-decaps \
  --batch 128 \
  --sk-in receiver_sk.demo_secret \
  --ct-in kem_ct.bin \
  --ss-out ss_receiver.demo_secret
```

正确性判断：

```bash
cmp ss_sender.demo_secret ss_receiver.demo_secret
```

如果两端 shared secret 一致，则 KEM 文件级接口正确。前端不会直接把 shared secret 当明文密钥使用，而是执行：

```text
AES-256-GCM key = SHA-256(shared secret)
```

随后用该 AES key 对文件内容进行加密和解密。

## 3. SIG 文件级接口

可执行文件：

```text
mldsaandaigis-sig/mldsa65_amd
```

当前开发包中如果目录仍为 `amd_sig_anchor_results_20260605_031411`，最终改名为 `mldsaandaigis-sig` 后，需要同步更新前端后端路径。

签名：

```bash
./mldsa65_amd \
  --api-sig-sign \
  --batch 128 \
  --msg-in manifest.payload.json \
  --pk-out sig_pk.bin \
  --sk-out sig_sk.demo_secret \
  --sig-out manifest.sig
```

验签：

```bash
./mldsa65_amd \
  --api-sig-verify \
  --batch 128 \
  --msg-in manifest.payload.json \
  --pk-in sig_pk.bin \
  --sig-in manifest.sig
```

前端中被签名的对象不是单个文件本身，而是 `manifest.payload.json`。该 payload 包含文件名、密文路径、nonce、tag、SHA-256 摘要、KEM ciphertext 路径和算法配置等信息。这样可以一次性保护整个传输包的结构和文件完整性。

## 4. 安全包关键文件

一次成功运行会生成类似结构：

```text
pack_xxx/
  manifest.json
  kem/
    kem_pk.bin
    kem_ct.bin
  sig/
    manifest.payload.json
    manifest.sig
    sig_pk.bin
  encrypted/
    *.enc
  recovered/
    ...
```

关键含义：

`manifest.json`：安全包主清单，记录算法配置、文件摘要、密文位置、KEM/SIG 后端信息和验证所需元数据。

`kem/kem_ct.bin`：KEM ciphertext，接收端使用私钥 decaps 后恢复 shared secret。

`sig/manifest.payload.json`：被 ML-DSA/Aigis-sig 签名的清单载荷。

`sig/manifest.sig`：manifest payload 的签名。

`encrypted/*.enc`：AES-256-GCM 加密后的文件密文。

`recovered/`：验证通过后恢复出的明文文件目录。

## 5. Batch/decomp 设计说明

本项目不强行使用单实例签名 CLI 作为主路径。原因是 ML-DSA/Aigis-sig 在 AMD ROCm 平台上存在更明显的资源压力，单实例或过重 kernel 容易受到 private segment、scratch、occupancy 等因素影响。

因此前端采用 batch/decomp 文件级接口作为实际应用路径：

```text
batch 提供 GPU 并行吞吐
decomp pipeline 降低单个签名路径的资源压力
文件级 API 让前端能够真实调用 KEM/SIG 能力
```

这也是项目的主要工程贡献之一：不是只给出 isolated benchmark，而是把 Kyber/Aigis-enc 和 ML-DSA/Aigis-sig 接成可演示、可验证、可扩展的 ROCm 后量子安全传输流程。
