# ML-DSA / Aigis-sig AMD 调试路线

当前阶段应先把签名部分调试扎实，再考虑 Kyber / Aigis-enc。KEM/ENC 可以作为独立子项目推进，不必现在强行同构；比赛材料里只需要把二者组织成同一套“后量子密码 GPU 加速系统”的两个模块即可。

## 目标顺序

1. 正确性稳定：六个参数集在 AMD W7900 上 `keygen + sign + verify + tamper-reject` 全部通过。
2. 性能基线完整：记录 batch=128/512/1024/2048/4096 的 latency、throughput。
3. 资源边界清楚：确认哪些 batch 或 kernel 会触发 HSA out of resources、private segment 过大、显存不足。
4. 瓶颈定位可解释：用程序内 `--profile` 和 `rocprofv3` 找到主要耗时 kernel。
5. 针对性优化：每次只改一个开关或一个 kernel，保留优化前后日志。

## 推荐执行流程

### 1. 构建

```bash
tar -xzf amd_upload.tar.gz
bash amd_tools/build_sig_amd.sh
```

构建失败时优先查：

- 是否设置 `--offload-arch=gfx1100`
- 是否设置 ROCm runtime 的 `LD_LIBRARY_PATH`
- 是否清除了 UTF-8 BOM
- 是否仍有 CUDA warp mask 的 32-bit 写法

### 2. 快速正确性矩阵

```bash
bash amd_tools/run_sig_debug_matrix.sh
```

这个脚本跑六个参数集的 batch=1/8/32/128。它比完整 sweep 快，适合每次源码变动后先跑一遍。输出：

```text
amd_results/debug/*.log
amd_results/sig_debug_summary.csv
```

若某个参数集失败，先固定一个最小失败 batch 单独跑：

```bash
./mldsa44_amd --batch 8 --quiet --skip-keygen-oracle
```

### 3. 完整 batch 曲线

```bash
bash amd_tools/run_sig_sweep.sh
```

输出：

```text
amd_results/sweep/*.log
amd_results/sig_sweep_summary.csv
```

论文/PPT 里至少整理这几列：

- 算法与参数集
- batch size
- Keygen latency / throughput
- Sign latency / throughput
- Verify latency / throughput
- Sign/Verify correctness
- 当前 sign path，例如 decomp pipeline

### 4. 单点 profiling

先选一个代表性组合，例如：

```bash
bash amd_tools/profile_sig_one.sh mldsa44_amd 1024
bash amd_tools/profile_sig_one.sh mldsa87_amd 1024
bash amd_tools/profile_sig_one.sh aigis2_amd 1024
```

重点看：

- `batch_sign_sample_y_kernel`
- `launch_batch_ntt` / `launch_batch_invntt`
- `batch_verify_matvec_kernel`
- `batch_sign_pointwise_cp_shared_kernel`
- pack/check 类 kernel
- 是否存在 private segment 明显偏大的 kernel

### 5. 优化实验顺序

建议按风险从低到高做：

1. 调 batch size：找每个参数集的吞吐峰值点和资源崩溃点。
2. 调 `BLOCK_SIZE`：现在 AMD 安全值是 `1`，可尝试 `2/4/8/16`，每次完整记录正确性和 HSA 错误。
3. 调 sign decomp 检查频率：尝试 `-DBATCH_SIGN_DECOMP_CHECK_INTERVAL=1/2/4/8`。
4. 只在小参数集上测试 tail fallback：大参数集先保持 `-DBATCH_SIGN_DECOMP_TAIL_ENABLE=0`。
5. 优化 matvec / NTT：这是论文里最容易讲清楚的核心 GPU 算子优化。
6. 再考虑 keygen sample split / pack fusion 等较细优化。

## 结论路线

现阶段不要把 Kyber / Aigis-enc 强行塞进签名框架。建议项目结构上保持：

```text
sig_amd/          ML-DSA + Aigis-sig AMD/HIP 同构签名框架
kem_enc_amd/      Kyber + Aigis-enc AMD/HIP 独立批处理框架
results/          统一放性能日志、CSV、profiling 结果
docs/             统一写 README、论文、PPT 图表
```

这样工程风险低，也更符合比赛评审关注点：真实 ROCm 使用、可复现数据、清晰 profiling 和针对性优化。
