# Kyber/Aigis-enc AMD 首轮跑通与瓶颈分析

日期：2026-06-10

## 1. 当前阶段结论

Kyber-768 已经在 AMD JupyterLab / ROCm 环境中完成首轮跑通。

当前已确认：

- `kyber768_amd` 可以通过 `hipcc` 编译生成。
- 运行时使用 HIP Runtime。
- 正确性测试通过，`KEM 正确性: PASS`。
- 已获得 AMD 单卡上的首批 keygen / encaps / decaps 吞吐数据。
- 已完成一次程序内 pipeline profile，定位到 keygen 的主要瓶颈。

这说明 Kyber/Aigis-enc 模块已经从“4090 上跑通”推进到“AMD ROCm 上可运行、可测量、可分析”的阶段。

## 2. AMD 环境与构建信息

AMD JupyterLab 中显示的设备信息：

```text
GPU: AMD Radeon Graphics (gfx1100, 48 CUs, 51.5 GB VRAM)
Runtime: HIP
Algorithm: Kyber-768  K=3  Q=3329
```

构建方式：

```bash
bash build_hip.sh kyber768
```

构建结果：

```text
84 warnings generated when compiling for host.
HIP 构建完成
```

说明：

- 当前 warning 暂时不影响可执行文件生成。
- 运行时需要设置 ROCm runtime 动态库路径，脚本中已内置：

```bash
export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"
```

## 3. 正确性冒烟结果

命令：

```bash
bash run_kem_smoke_amd.sh
cat amd_results/kem_smoke_summary.csv
```

结果摘要：

| Algorithm | Batch | Keygen ops/s | Encaps ops/s | Decaps ops/s | Correctness |
|---|---:|---:|---:|---:|---|
| Kyber-768 | 1 | 724 | 578 | 471 | PASS |
| Kyber-768 | 8 | 5,695 | 4,572 | 3,712 | PASS |
| Kyber-768 | 32 | 27,020 | 22,823 | 19,908 | PASS |
| Kyber-768 | 128 | 101,527 | 89,225 | 78,159 | PASS |

结论：

- 小 batch 下正确性稳定。
- 随 batch 增大，吞吐明显提升，说明该工作负载具备 GPU 批处理收益。
- 小 batch 的吞吐较低，后续可作为 kernel launch 开销和任务粒度不足的分析案例。

## 3.1 全量 KEM 目标冒烟结果

后续已完成 7 个 KEM 目标的 AMD 小 batch 冒烟测试：

```bash
bash build_hip.sh
bash run_kem_smoke_amd.sh
cat amd_results/kem_smoke_summary.csv
```

结果：

| Target | K | Q | Batch=128 Keygen | Batch=128 Encaps | Batch=128 Decaps | Correctness |
|---|---:|---:|---:|---:|---:|---|
| Kyber-512 | 2 | 3329 | 156,364 ops/s | 142,702 ops/s | 122,041 ops/s | PASS |
| Kyber-768 | 3 | 3329 | 101,685 ops/s | 88,712 ops/s | 77,192 ops/s | PASS |
| Kyber-1024 | 4 | 3329 | 63,358 ops/s | 66,156 ops/s | 61,995 ops/s | PASS |
| Aigis-enc-1 | 2 | 7681 | 94,551 ops/s | 70,573 ops/s | 51,935 ops/s | PASS |
| Aigis-enc-2 | 3 | 7681 | 65,397 ops/s | 49,648 ops/s | 39,623 ops/s | PASS |
| Aigis-enc-3 | 3 | 7681 | 60,851 ops/s | 49,192 ops/s | 40,077 ops/s | PASS |
| Aigis-enc-4 | 4 | 7681 | 43,553 ops/s | 36,797 ops/s | 32,014 ops/s | PASS |

阶段性结论：

> Kyber-512/768/1024 与 Aigis-enc-1/2/3/4 均已在 AMD ROCm 上完成小 batch 正确性验证，说明 KEM 模块的 AMD 基础迁移已经闭环。后续工作重点应从“能否运行”转向“最佳 batch 吞吐、热点瓶颈定位与 ROCm 专项优化”。

可用于 PPT 的表述：

> 本项目已完成 Kyber 与 Aigis-enc 共 7 个参数集在 AMD ROCm 平台上的编译、运行与正确性冒烟验证，为后续构建完整后量子科研数据可信流转平台提供了可复现的 KEM 基础能力。

## 4. 当前最好性能

命令：

```bash
cat amd_results/kem_best.csv
```

当前 Kyber-768 最好结果：

| Operation | Best Throughput | Best Config |
|---|---:|---|
| Keygen | 4,521,348 ops/s | batch=32768, serial, streams=1 |
| Encaps | 5,932,625 ops/s | batch=32768, serial, streams=1 |
| Decaps | 5,509,231 ops/s | batch=32768, serial, streams=1 |

与此前 4090 Kyber-768 最好结果对比：

| Platform | Keygen | Encaps | Decaps |
|---|---:|---:|---:|
| RTX 4090D | 5.70M ops/s | 7.95M ops/s | 8.16M ops/s |
| AMD 初始 ROCm 版 | 4.52M ops/s | 5.93M ops/s | 5.51M ops/s |

阶段性判断：

> 初始 HIP/ROCm 迁移版已经达到 RTX 4090D 同量级吞吐。当前性能差距不能简单归因于 AMD 硬件不足，更合理的解释是：现有实现仍是迁移与初步适配版本，尚未充分围绕 ROCm/RDNA3 的执行模型进行采样、访存和并行粒度调优。

## 4.1 全量 KEM Sweep 最佳性能表

命令：

```bash
bash run_kem_sweep_amd.sh
cat amd_results/kem_best.csv
```

AMD ROCm 全量 KEM 最佳结果：

| Algorithm | Best Keygen | Keygen Config | Best Encaps | Encaps Config | Best Decaps | Decaps Config |
|---|---:|---|---:|---|---:|---|
| Kyber-512 | 6,553,707 ops/s | batch=32768 serial | 10,898,867 ops/s | batch=65536 serial | 7,200,522 ops/s | batch=32768 serial |
| Kyber-768 | 4,797,831 ops/s | batch=32768 serial | 5,893,981 ops/s | batch=32768 serial | 5,502,606 ops/s | batch=32768 serial |
| Kyber-1024 | 3,691,492 ops/s | batch=32768 serial | 4,207,232 ops/s | batch=32768 serial | 3,674,413 ops/s | batch=32768 serial |
| Aigis-enc-1 | 8,316,531 ops/s | batch=65536 serial | 7,092,336 ops/s | batch=65536 serial | 5,175,258 ops/s | batch=65536 serial |
| Aigis-enc-2 | 5,708,114 ops/s | batch=65536 serial | 4,656,565 ops/s | batch=65536 serial | 3,360,105 ops/s | batch=65536 serial |
| Aigis-enc-3 | 5,159,607 ops/s | batch=65536 serial | 4,589,059 ops/s | batch=65536 serial | 3,583,328 ops/s | batch=65536 serial |
| Aigis-enc-4 | 3,874,049 ops/s | batch=65536 serial | 2,943,683 ops/s | batch=65536 serial | 2,370,340 ops/s | batch=65536 serial |

阶段性结论：

- Kyber 系列最佳 batch 多集中在 `32768`，Kyber-512 的 encaps 在 `65536` 达到最高。
- Aigis-enc 系列最佳 batch 均集中在 `65536`，说明 Aigis-enc 在当前实现下更依赖大 batch 来摊薄 launch 和调度开销。
- 当前 AMD 初始版已经具备百万级到千万级 KEM 吞吐，足以支撑“多文件科研数据可信流转”的高并发密钥封装场景。
- 高安全等级参数集吞吐下降明显，后续可将 Kyber-1024 和 Aigis-enc-4 作为重点 profile 对象。

适合 PPT 的结论：

> 在 AMD ROCm 初始适配版本中，Kyber-512 encaps 已达到 10.90M ops/s，Aigis-enc-1 keygen 达到 8.32M ops/s。全量 7 个 KEM 参数集均达到百万级以上吞吐，证明 AMD Radeon PRO/RDNA3 平台具备支撑高并发后量子密钥封装的工程潜力。

## 5. Pipeline Profile 结果

命令：

```bash
bash profile_kem_one_amd.sh kyber768_amd 32768 3
cat amd_results/profile/kyber768_amd_b32768_profile.log
```

profile 输出：

```text
--- batch=32768 n_ops=3 mode=serial ---
  Keygen:      7.3 ms/batch -> 4512026 ops/sec
  Encaps:      5.5 ms/batch -> 5977275 ops/sec
  Decaps:      6.0 ms/batch -> 5488131 ops/sec

--- batch=32768 n_ops=3 mode=pipeline ---
  Pipeline profile: sample=5.065 ntt=0.454 matvec=0.487 invntt=0.349 add=0.315 pack=0.569 total=7.238 ms
  Keygen:      7.3 ms/batch -> 4517392 ops/sec
  Encaps:      4.6 ms/batch -> 7149509 ops/sec
  Decaps:      5.8 ms/batch -> 5632301 ops/sec
```

阶段占比估算：

| Stage | Time | Ratio |
|---|---:|---:|
| sample | 5.065 ms | 70.0% |
| ntt | 0.454 ms | 6.3% |
| matvec | 0.487 ms | 6.7% |
| invntt | 0.349 ms | 4.8% |
| add | 0.315 ms | 4.4% |
| pack | 0.569 ms | 7.9% |
| total | 7.238 ms | 100% |

核心瓶颈：

> 当前 Kyber-768 keygen 的主要瓶颈是 sample 阶段，占 pipeline 总耗时约 70%。这说明在 AMD ROCm 初始实现中，主要限制来自 SHAKE/XOF 展开、拒绝采样和噪声采样相关路径，而不是 NTT 或矩阵向量乘。

## 5.1 多目标 Profile 对比

已完成三个代表目标的 profile：

```bash
bash profile_kem_one_amd.sh kyber768_amd 32768 3
bash profile_kem_one_amd.sh kyber1024_amd 32768 3
bash profile_kem_one_amd.sh aigisenc4_amd 32768 3
```

### Kyber-768

```text
Pipeline profile: sample=5.075 ntt=0.453 matvec=0.487 invntt=0.346 add=0.315 pack=0.572 total=7.247 ms
```

| Stage | Time | Ratio |
|---|---:|---:|
| sample | 5.075 ms | 70.0% |
| ntt | 0.453 ms | 6.3% |
| matvec | 0.487 ms | 6.7% |
| invntt | 0.346 ms | 4.8% |
| add | 0.315 ms | 4.3% |
| pack | 0.572 ms | 7.9% |

### Kyber-1024

```text
Pipeline profile: sample=7.227 ntt=0.628 matvec=0.887 invntt=0.477 add=0.437 pack=0.930 total=10.584 ms
```

| Stage | Time | Ratio |
|---|---:|---:|
| sample | 7.227 ms | 68.3% |
| ntt | 0.628 ms | 5.9% |
| matvec | 0.887 ms | 8.4% |
| invntt | 0.477 ms | 4.5% |
| add | 0.437 ms | 4.1% |
| pack | 0.930 ms | 8.8% |

### Aigis-enc-4

```text
Pipeline profile: sample=9.439 ntt=0.840 matvec=0.811 invntt=0.470 add=0.438 pack=0.781 total=12.780 ms
```

| Stage | Time | Ratio |
|---|---:|---:|
| sample | 9.439 ms | 73.9% |
| ntt | 0.840 ms | 6.6% |
| matvec | 0.811 ms | 6.3% |
| invntt | 0.470 ms | 3.7% |
| add | 0.438 ms | 3.4% |
| pack | 0.781 ms | 6.1% |

综合判断：

> Kyber-768、Kyber-1024 和 Aigis-enc-4 均表现出 sample 阶段主导的瓶颈特征，占 keygen pipeline 总耗时约 68% 到 74%。因此，下一阶段优化不应优先放在 NTT 或矩阵向量乘，而应优先围绕 SHAKE/XOF 展开、拒绝采样、噪声采样以及采样阶段的并行粒度进行 ROCm 专项调优。

另一个重要现象：

| Target | Serial Keygen | Pipeline Keygen | 现象 |
|---|---:|---:|---|
| Kyber-768 | 4.20M ops/s | 4.52M ops/s | pipeline 略优 |
| Kyber-1024 | 3.68M ops/s | 3.16M ops/s | pipeline 反而变慢 |
| Aigis-enc-4 | 3.44M ops/s | 2.57M ops/s | pipeline 明显变慢 |

说明：

> 当前 pipeline keygen 不是所有参数集的最优路径。对于 Kyber-1024 和 Aigis-enc-4，大参数集下 pipeline 引入的中间缓冲、拆分 kernel 和打包路径开销超过了收益。因此短期 benchmark 应继续以 serial 路径作为 best-throughput 基线，同时将 pipeline 路径作为 profile 和优化实验对象。

## 6. 当前暴露的问题

### 6.1 简单迁移能跑，但不代表跑满

当前结果说明 HIP 迁移已经完成基本功能，但 profile 显示 sample 阶段占比过高。后续需要针对 AMD/RDNA3 重新调整采样并行粒度、访存布局和 kernel 拆分方式。

适合论文表述：

> HIP 迁移解决了代码在 ROCm 平台上的可运行性问题，但后量子密码算法中的采样和 XOF 扩展并不是 ROCm 现有 AI/HPC 库重点覆盖的算子类型。简单迁移不能充分发挥 AMD GPU 的潜力，需要结合 profile 数据进行面向 ROCm 的专门优化。

### 6.2 ROCm 缺少类 cuPQC 的后量子密码专用库

NVIDIA 已经有 cuPQC，面向 ML-KEM、ML-DSA 等后量子密码负载提供 GPU 库支持。AMD ROCm 目前公开生态中缺少同类 PQC 专用库。

本项目可作为 AMD 生态补齐方向：

- 批量 ML-KEM/Kyber KEM kernel。
- 批量 Aigis-enc KEM kernel。
- 批量 ML-DSA/Aigis-sig 签名验签 kernel。
- NTT、Keccak/SHAKE、采样、packing 等 PQC 基础算子。
- 面向多文件科研数据可信流转的上层接口。

### 6.3 小 batch 端到端效率低

batch=1/8/32 时吞吐明显低于大 batch。这说明小任务场景下 GPU launch、同步和数据准备开销占比较高。

后续优化方向：

- 多文件任务聚合。
- 自适应 batch size。
- 多 stream 并发。
- 缓冲区复用。
- CPU I/O 与 GPU 密码计算流水线化。

## 7. 下一步实验计划

### Step 1：验证 split sampling 是否改善 sample 瓶颈

目的：

当前 sample 阶段占总耗时约 70%，优先测试已有的 `KEM_SPLIT_KEYGEN_SAMPLE=1` 路径。

在 AMD JupyterLab 中执行：

```bash
cd /app/kyberandaigis-enc
export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

hipcc -O2 -std=c++17 -x hip --offload-arch=gfx1100 \
  -DKEM_SERIAL_TPB=64 \
  -DKEM_SPLIT_KEYGEN_SAMPLE=1 \
  -DALGORITHM=1 -DPARAM_MODE=3 \
  main.cu -o kyber768_split_amd

./kyber768_split_amd --batch 32768 --n-ops 3 --no-correctness --pipeline --profile-pipeline
./kyber768_split_amd --batch 128 --n-ops 1
```

需要记录：

- `sample` 时间是否下降。
- `total` 时间是否下降。
- `Keygen ops/s` 是否提升。
- 正确性是否仍然 PASS。

如果 split 版本有效，可写成第一个优化点：

> 通过拆分 seed expand、matrix sample、noise sample，提高采样阶段并行粒度，降低 sample 阶段耗时。

### Step 1 实验结果：split sampling 失败

实际执行：

```bash
hipcc -O2 -std=c++17 -x hip --offload-arch=gfx1100 \
  -DKEM_SERIAL_TPB=64 \
  -DKEM_SPLIT_KEYGEN_SAMPLE=1 \
  -DALGORITHM=1 -DPARAM_MODE=3 \
  main.cu -o kyber768_split_amd

./kyber768_split_amd --batch 32768 --n-ops 3 --no-correctness --pipeline --profile-pipeline
./kyber768_split_amd --batch 128 --n-ops 1
```

结果：

```text
--- batch=32768 n_ops=3 mode=serial ---
  Keygen:      7.2 ms/batch -> 4556756 ops/sec
  Encaps:      5.6 ms/batch -> 5883818 ops/sec
  Decaps:      6.0 ms/batch -> 5502531 ops/sec

--- batch=32768 n_ops=3 mode=pipeline ---
  Pipeline profile: sample=16.648 ntt=0.476 matvec=0.523 invntt=0.346 add=0.318 pack=0.571 total=18.883 ms
  Keygen:     18.8 ms/batch -> 1742529 ops/sec
  Encaps:      4.6 ms/batch -> 7100841 ops/sec
  Decaps:      5.8 ms/batch -> 5633798 ops/sec
```

正确性：

```text
KEM 正确性: PASS
```

对比：

| Version | sample | total | Pipeline Keygen |
|---|---:|---:|---:|
| 原 pipeline | 5.075 ms | 7.247 ms | 4.52M ops/s |
| split sample | 16.648 ms | 18.883 ms | 1.74M ops/s |

结论：

> `KEM_SPLIT_KEYGEN_SAMPLE=1` 在 AMD ROCm 上不是有效优化。简单拆分 seed expand、matrix sample、noise sample 会显著增加 sample 阶段耗时，推测原因是额外 kernel launch、全局内存中间结果写回、cache/locality 下降和更高的调度开销超过了并行粒度提升带来的收益。

论文中可以将该实验作为“负优化案例”：

> 并非所有 CUDA/NVIDIA 风格或直觉上的 kernel 拆分都适合 AMD ROCm。对后量子密码采样路径而言，kernel fusion 与数据局部性可能比盲目拆分更重要。该实验说明需要 profile-driven tuning，而不是仅通过增加 kernel 数量提高表面并行度。

下一步调整：

> 放弃 split sampling 路线，优先保留当前 serial/baseline 路径作为最佳吞吐实现；后续优化转向 `KEM_SERIAL_TPB`、batch size、stream 并发、采样 kernel 内部并行方式和减少中间内存访问。

### Step 2 实验结果：TPB=256 对 Kyber-768 有效

在 Kyber-768 上测试 `KEM_SERIAL_TPB=256`：

```bash
hipcc -O2 -std=c++17 -x hip --offload-arch=gfx1100 \
  -DKEM_SERIAL_TPB=256 \
  -DALGORITHM=1 -DPARAM_MODE=3 \
  main.cu -o kyber768_tpb256_amd

./kyber768_tpb256_amd --batch 32768 --n-ops 5 --no-correctness
```

结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.0 ms/batch -> 5444673 ops/sec
  Encaps:      5.4 ms/batch -> 6035024 ops/sec
  Decaps:      5.9 ms/batch -> 5581568 ops/sec
```

与此前 Kyber-768 sweep 最佳值对比：

| Metric | 原最佳 | TPB=256 | 变化 |
|---|---:|---:|---:|
| Keygen | 4,797,831 ops/s | 5,444,673 ops/s | +13.5% |
| Encaps | 5,893,981 ops/s | 6,035,024 ops/s | +2.4% |
| Decaps | 5,502,606 ops/s | 5,581,568 ops/s | +1.4% |

结论：

> 相比 split sampling，`KEM_SERIAL_TPB` 调参是当前更有效的 ROCm 优化方向。将 Kyber-768 的 serial kernel 线程块大小从默认 64 调整到 256 后，keygen 吞吐提升约 13.5%，说明 AMD/RDNA3 上的线程组织参数对后量子 KEM 批处理性能有显著影响。

下一步需要补充：

- 补齐 `TPB=32/64/128/256/512` 的完整对比。
- 对 `TPB=256` 跑一次正确性测试。
- 若 `TPB=256` 稳定，再推广到 Kyber-512/1024 与 Aigis-enc 系列。

### Step 2 补充：TPB=512 不是最优

`KEM_SERIAL_TPB=512` 正确性测试：

```text
KEM 正确性: PASS
```

性能结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.5 ms/batch -> 5049880 ops/sec
  Encaps:      5.8 ms/batch -> 5603379 ops/sec
  Decaps:      6.3 ms/batch -> 5232946 ops/sec
```

与 `TPB=256` 对比：

| Metric | TPB=256 | TPB=512 | 结论 |
|---|---:|---:|---|
| Keygen | 5,444,673 ops/s | 5,049,880 ops/s | TPB=512 下降 |
| Encaps | 6,035,024 ops/s | 5,603,379 ops/s | TPB=512 下降 |
| Decaps | 5,581,568 ops/s | 5,232,946 ops/s | TPB=512 下降 |

判断：

> `TPB=512` 虽然正确性通过，但性能低于 `TPB=256`。说明继续增大线程块并不能进一步提升吞吐，可能引入更高的寄存器/调度压力或降低有效 occupancy。当前 Kyber-768 的候选最优配置仍为 `KEM_SERIAL_TPB=256`。

### Step 2 补充：TPB=128 结果

`KEM_SERIAL_TPB=128` 正确性测试：

```text
KEM 正确性: PASS
```

性能结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.4 ms/batch -> 5081079 ops/sec
  Encaps:      5.6 ms/batch -> 5885813 ops/sec
  Decaps:      5.9 ms/batch -> 5526684 ops/sec
```

当前已知 TPB 对比：

| TPB | Correctness | Keygen | Encaps | Decaps |
|---:|---|---:|---:|---:|
| 32 | PASS | 4,909,746 ops/s | 5,984,596 ops/s | 5,450,378 ops/s |
| 64 | PASS | 4,866,205 ops/s | 5,994,111 ops/s | 5,544,629 ops/s |
| 128 | PASS | 5,081,079 ops/s | 5,885,813 ops/s | 5,526,684 ops/s |
| 256 | PASS | 5,444,673 ops/s | 6,035,024 ops/s | 5,581,568 ops/s |
| 512 | PASS | 5,049,880 ops/s | 5,603,379 ops/s | 5,232,946 ops/s |

阶段判断：

> 在已测试的 32/64/128/256/512 中，`TPB=256` 对 Kyber-768 的 keygen、encaps、decaps 均为当前最好。相较默认 `TPB=64`，`TPB=256` 使 keygen 从 4,866,205 ops/s 提升到 5,444,673 ops/s，提升约 11.9%。该结果支持将 `KEM_SERIAL_TPB=256` 作为 AMD/RDNA3 上 Kyber-768 serial 路径的候选默认配置。

### Step 2 补充：TPB=64 默认基线

`KEM_SERIAL_TPB=64` 正确性测试：

```text
KEM 正确性: PASS
```

性能结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.7 ms/batch -> 4866205 ops/sec
  Encaps:      5.5 ms/batch -> 5994111 ops/sec
  Decaps:      5.9 ms/batch -> 5544629 ops/sec
```

与 `TPB=256` 对比：

| Metric | TPB=64 | TPB=256 | 变化 |
|---|---:|---:|---:|
| Keygen | 4,866,205 ops/s | 5,444,673 ops/s | +11.9% |
| Encaps | 5,994,111 ops/s | 6,035,024 ops/s | +0.7% |
| Decaps | 5,544,629 ops/s | 5,581,568 ops/s | +0.7% |

结论：

> `TPB=256` 的主要收益集中在 keygen，encaps/decaps 提升较小。这说明 keygen 路径对线程块大小和调度粒度更敏感，后续优化仍应围绕 keygen 的采样/密钥生成路径展开。

### Step 2 补充：TPB=32 结果

`KEM_SERIAL_TPB=32` 正确性测试：

```text
KEM 正确性: PASS
```

性能结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.7 ms/batch -> 4909746 ops/sec
  Encaps:      5.5 ms/batch -> 5984596 ops/sec
  Decaps:      6.0 ms/batch -> 5450378 ops/sec
```

结论：

> `TPB=32` 正确性通过，但 keygen、encaps、decaps 均未超过 `TPB=256`。至此 Kyber-768 的 TPB sweep 闭合，`TPB=256` 是当前最优线程块大小。

### Step 3 初步推广：Aigis-enc-4 上 TPB=256 不是全局最优

对 Aigis-enc-4 使用 `KEM_SERIAL_TPB=256`，batch=65536：

```text
--- batch=65536 n_ops=5 mode=serial ---
  Keygen:     16.8 ms/batch -> 3908677 ops/sec
  Encaps:     23.1 ms/batch -> 2837074 ops/sec
  Decaps:     28.2 ms/batch -> 2325429 ops/sec
```

与全量 sweep 中 Aigis-enc-4 原最佳对比：

| Metric | 原最佳 | TPB=256 | 变化 |
|---|---:|---:|---:|
| Keygen | 3,874,049 ops/s | 3,908,677 ops/s | +0.9% |
| Encaps | 2,943,683 ops/s | 2,837,074 ops/s | -3.6% |
| Decaps | 2,370,340 ops/s | 2,325,429 ops/s | -1.9% |

判断：

> `TPB=256` 对 Aigis-enc-4 的 keygen 只有轻微收益，并降低 encaps/decaps 吞吐。因此 `TPB=256` 不应直接作为所有 KEM 算法的全局默认值。更合理的策略是按算法和操作类型选择配置：Kyber-768 可优先采用 `TPB=256`，Aigis-enc-4 暂时保留原配置作为总体吞吐基线。

### Step 3 补充：Kyber-1024 与 Aigis-enc-1 的 TPB=256 推广

#### Kyber-1024

`KEM_SERIAL_TPB=256`，batch=32768：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      8.2 ms/batch -> 3975785 ops/sec
  Encaps:      7.6 ms/batch -> 4295105 ops/sec
  Decaps:      8.8 ms/batch -> 3707815 ops/sec
```

与全量 sweep 中 Kyber-1024 原最佳对比：

| Metric | 原最佳 | TPB=256 | 变化 |
|---|---:|---:|---:|
| Keygen | 3,691,492 ops/s | 3,975,785 ops/s | +7.7% |
| Encaps | 4,207,232 ops/s | 4,295,105 ops/s | +2.1% |
| Decaps | 3,674,413 ops/s | 3,707,815 ops/s | +0.9% |

判断：

> `TPB=256` 对 Kyber-1024 同样有效，尤其 keygen 提升约 7.7%。这说明 Kyber 系列在 AMD/RDNA3 上普遍受益于更大的 serial kernel 线程块配置。

#### Aigis-enc-1

`KEM_SERIAL_TPB=256`，batch=65536：

```text
--- batch=65536 n_ops=5 mode=serial ---
  Keygen:      7.2 ms/batch -> 9113161 ops/sec
  Encaps:      9.5 ms/batch -> 6890204 ops/sec
  Decaps:     12.7 ms/batch -> 5158510 ops/sec
```

与全量 sweep 中 Aigis-enc-1 原最佳对比：

| Metric | 原最佳 | TPB=256 | 变化 |
|---|---:|---:|---:|
| Keygen | 8,316,531 ops/s | 9,113,161 ops/s | +9.6% |
| Encaps | 7,092,336 ops/s | 6,890,204 ops/s | -2.9% |
| Decaps | 5,175,258 ops/s | 5,158,510 ops/s | -0.3% |

判断：

> `TPB=256` 对 Aigis-enc-1 的 keygen 也有明显收益，但会降低 encaps，decaps 基本持平。这进一步说明三类 KEM 操作对线程块大小的敏感性不同，后续不应继续用单一 `KEM_SERIAL_TPB` 控制 keygen、encaps 和 decaps。

### Step 4 新优化方向：按操作拆分 TPB

当前证据：

- Kyber-768：`TPB=256` 使 keygen 提升约 11.9%。
- Kyber-1024：`TPB=256` 使 keygen 提升约 7.7%。
- Aigis-enc-1：`TPB=256` 使 keygen 提升约 9.6%，但 encaps 下降约 2.9%。
- Aigis-enc-4：`TPB=256` 使 keygen 轻微提升约 0.9%，但 encaps/decaps 下降。

结论：

> 下一步应将当前单一 `KEM_SERIAL_TPB` 拆分为 `KEM_KEYGEN_TPB`、`KEM_ENCAPS_TPB`、`KEM_DECAPS_TPB`。这样可以让 keygen 使用更适合 AMD 的 `TPB=256`，同时让 encaps/decaps 继续保留各自更优的配置，避免一个编译宏同时影响三种不同操作。

### Step 4 初测：按操作拆分 TPB

代码已将单一 `KEM_SERIAL_TPB` 拆分为：

```c
KEM_KEYGEN_TPB
KEM_ENCAPS_TPB
KEM_DECAPS_TPB
```

测试配置：

```bash
KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=64 KEM_DECAPS_TPB=64 bash build_hip.sh kyber768
./kyber768_amd --batch 128 --n-ops 1
./kyber768_amd --batch 32768 --n-ops 5 --no-correctness
```

正确性：

```text
KEM 正确性: PASS
```

性能结果：

```text
--- batch=32768 n_ops=5 mode=serial ---
  Keygen:      6.3 ms/batch -> 5223903 ops/sec
  Encaps:      5.5 ms/batch -> 5948418 ops/sec
  Decaps:      5.9 ms/batch -> 5568138 ops/sec
```

与默认 `TPB=64/64/64` 对比：

| Metric | 默认 64/64/64 | 拆分 256/64/64 | 变化 |
|---|---:|---:|---:|
| Keygen | 4,866,205 ops/s | 5,223,903 ops/s | +7.4% |
| Encaps | 5,994,111 ops/s | 5,948,418 ops/s | -0.8% |
| Decaps | 5,544,629 ops/s | 5,568,138 ops/s | +0.4% |

阶段判断：

> 按操作拆分 TPB 是有效方向。`256/64/64` 组合在基本不损失 encaps/decaps 的情况下，将 Kyber-768 keygen 较默认配置提升约 7.4%。但该结果低于此前全 `TPB=256` 单次测试中的 5.44M keygen，因此需要用更多迭代次数复测，排除运行波动影响。

### Step 4 复测：n_ops=20 稳定态结果

为降低短迭代测量波动，将 `n_ops` 从 5 提高到 20 后重新测试 Kyber-768，batch=32768。

结果：

| Config | Keygen | Encaps | Decaps |
|---|---:|---:|---:|
| 256/64/64 | 6,190,326 ops/s | 6,023,175 ops/s | 5,610,136 ops/s |
| 256/256/256 | 6,226,843 ops/s | 6,026,556 ops/s | 5,612,435 ops/s |
| 256/128/128 | 6,297,089 ops/s | 6,026,840 ops/s | 5,617,614 ops/s |

阶段性结论：

> `n_ops=20` 复测表明，Kyber-768 在 AMD 上的稳定态 keygen 吞吐可达到 6.2M ops/s 以上，显著高于早期 `n_ops=5` 的 5.2M 到 5.4M 结果。三种 TPB 组合的 encaps/decaps 基本一致，说明当前实现中 keygen 对 TPB 更敏感，而 encaps/decaps 对 64/128/256 的差异不明显。

重要修正：

> 后续论文中的性能结论应优先采用较大迭代次数的稳定态结果，而不是早期 `n_ops=3/5` 的短迭代结果。短迭代更适合快速调试，稳定性能表建议统一使用 `n_ops=20` 或更高。

## 12. Kyber 系列 n_ops=20 稳定性能表

使用配置：

```text
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
batch=32768
n_ops=20
```

命令：

```bash
mkdir -p amd_results/final_nops20

for target in kyber512 kyber768 kyber1024; do
  KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 \
  bash build_hip.sh "$target" 2>&1 | tee -a amd_results/final_nops20/kyber_nops20.log

  ./${target}_amd --batch 32768 --n-ops 20 --no-correctness \
    2>&1 | tee -a amd_results/final_nops20/kyber_nops20.log
done
```

稳定性能结果：

| Algorithm | Keygen | Encaps | Decaps | Batch | Iterations |
|---|---:|---:|---:|---:|---:|
| Kyber-512 | 10,538,121 ops/s | 11,366,292 ops/s | 7,442,359 ops/s | 32768 | 20 |
| Kyber-768 | 6,184,066 ops/s | 6,018,408 ops/s | 5,623,416 ops/s | 32768 | 20 |
| Kyber-1024 | 4,665,935 ops/s | 4,277,726 ops/s | 3,860,981 ops/s | 32768 | 20 |

与早期短迭代 sweep 相比：

- Kyber-512 keygen 从 6.55M 提升到 10.54M。
- Kyber-768 keygen 从 4.80M 提升到 6.18M。
- Kyber-1024 keygen 从 3.69M 提升到 4.67M。

阶段性结论：

> 使用较大迭代次数与按操作拆分 TPB 后，AMD ROCm 上 Kyber 系列稳定态吞吐显著提升。Kyber-768 keygen 达到 6.18M ops/s，已经超过此前 RTX 4090D 记录中的 5.70M ops/s；Kyber-512 encaps 达到 11.37M ops/s，达到千万级后量子 KEM 吞吐。

适合论文/PPT 表述：

> 通过 ROCm 环境适配、正确性验证、profile 定位与 TPB 调优，Kyber 系列在 AMD Radeon/RDNA3 平台上达到百万级到千万级稳定吞吐，证明 AMD GPU 在后量子密钥封装批处理场景中具备与高端 NVIDIA GPU 同量级的性能竞争力。

## 13. Aigis-enc 系列 n_ops=20 稳定性能表

使用配置：

```text
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
batch=65536
n_ops=20
```

命令：

```bash
mkdir -p amd_results/final_nops20

for target in aigisenc1 aigisenc2 aigisenc3 aigisenc4; do
  KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 \
  bash build_hip.sh "$target" 2>&1 | tee -a amd_results/final_nops20/aigisenc_nops20.log

  ./${target}_amd --batch 65536 --n-ops 20 --no-correctness \
    2>&1 | tee -a amd_results/final_nops20/aigisenc_nops20.log
done
```

稳定性能结果：

| Algorithm | Keygen | Encaps | Decaps | Batch | Iterations |
|---|---:|---:|---:|---:|---:|
| Aigis-enc-1 | 10,368,547 ops/s | 7,200,232 ops/s | 5,200,585 ops/s | 65536 | 20 |
| Aigis-enc-2 | 6,879,027 ops/s | 4,701,072 ops/s | 3,370,967 ops/s | 65536 | 20 |
| Aigis-enc-3 | 6,344,582 ops/s | 4,629,810 ops/s | 3,594,671 ops/s | 65536 | 20 |
| Aigis-enc-4 | 4,261,266 ops/s | 2,955,561 ops/s | 2,425,389 ops/s | 65536 | 20 |

与早期短迭代 sweep 相比：

- Aigis-enc-1 keygen 从 8.32M 提升到 10.37M。
- Aigis-enc-2 keygen 从 5.71M 提升到 6.88M。
- Aigis-enc-3 keygen 从 5.16M 提升到 6.34M。
- Aigis-enc-4 keygen 从 3.87M 提升到 4.26M。

阶段性结论：

> Aigis-enc 系列在 AMD ROCm 上同样达到稳定百万级到千万级吞吐。Aigis-enc-1 keygen 达到 10.37M ops/s，说明 AMD 平台不仅能够支撑标准 Kyber/ML-KEM 类负载，也能支撑国产 Aigis-enc 系列后量子密钥封装算法的高并发批处理。

## 14. KEM 模块当前最终成果

截至当前阶段，Kyber/Aigis-enc KEM 模块已经完成：

- 7 个目标全部在 AMD ROCm 上编译通过。
- 7 个目标小 batch 正确性全部 PASS。
- 完成 Kyber-768、Kyber-1024、Aigis-enc-4 的程序内 profile。
- 完成 Kyber-768 的 `rocprofv3` kernel trace。
- 定位 sample/XOF/rejection sampling 为 keygen pipeline 主要瓶颈。
- 验证 split sampling 是负优化。
- 验证 TPB sweep 与按操作拆分 TPB 是有效调优方向。
- 形成 `n_ops=20` 稳定性能表。

当前最佳稳定结果摘要：

| Family | Representative Best |
|---|---|
| Kyber | Kyber-512 encaps 11.37M ops/s |
| Kyber | Kyber-768 keygen 6.18M ops/s |
| Kyber | Kyber-1024 keygen 4.67M ops/s |
| Aigis-enc | Aigis-enc-1 keygen 10.37M ops/s |
| Aigis-enc | Aigis-enc-4 keygen 4.26M ops/s |

可写入论文的阶段性总述：

> 本项目在 AMD ROCm/RDNA3 平台上完成 Kyber 与 Aigis-enc 共 7 个参数集的批量 KEM 实现验证。通过 ROCm 工具与程序内 profile 定位，发现 keygen 路径主要瓶颈集中在 sample/XOF/rejection sampling 阶段；通过 TPB sweep 和按操作拆分 kernel launch 配置，获得稳定百万级到千万级吞吐。其中 Kyber-512 encaps 达到 11.37M ops/s，Aigis-enc-1 keygen 达到 10.37M ops/s，证明 AMD 平台在后量子密钥封装批处理场景中具备较强工程竞争力。

## 16. 最终自动化报告记录

最终报告脚本：

```bash
bash run_kem_final_report_amd.sh
bash run_kem_resource_profile_amd.sh kyber768 32768 200
```

生成目录：

```text
amd_results/final_report_20260612_074154
amd_results/resource_profile_kyber768_20260612_074231
```

最终报告提取文件：

```text
amd_results/final_report_20260612_074154/kem_final_extract.txt
```

最终 KEM 性能表：

| Algorithm | Keygen | Encaps | Decaps |
|---|---:|---:|---:|
| Kyber-512 | 10,132,546 ops/s | 11,307,354 ops/s | 7,495,810 ops/s |
| Kyber-768 | 6,331,652 ops/s | 5,998,665 ops/s | 5,659,745 ops/s |
| Kyber-1024 | 4,484,088 ops/s | 4,290,608 ops/s | 3,836,327 ops/s |
| Aigis-enc-1 | 10,326,268 ops/s | 7,204,754 ops/s | 5,202,343 ops/s |
| Aigis-enc-2 | 6,639,865 ops/s | 4,704,347 ops/s | 3,367,005 ops/s |
| Aigis-enc-3 | 6,361,075 ops/s | 4,625,470 ops/s | 3,595,634 ops/s |
| Aigis-enc-4 | 4,144,860 ops/s | 2,951,076 ops/s | 2,429,642 ops/s |

资源/profile 报告目录：

```text
amd_results/resource_profile_kyber768_20260612_074231
```

该目录包含：

```text
metadata.txt
build.log
benchmark.log
rocm_smi_during.log
rocm_smi_gpu0_extract.log
rocprofv3/
rocprofv3_summary.csv
```

说明：

> `rocprofv3` 运行会引入额外开销，因此 profile 目录中的 `n_ops=1` 吞吐只用于定位 kernel 和 API 行为，不作为最终性能数据。最终性能应引用 `final_report_20260612_074154/kem_final_extract.txt` 中的 `n_ops=20` 稳定结果。

## 9. TPB 是什么，以及为什么要调

`TPB` 是 `threads per block`，表示每个 GPU kernel 线程块中包含多少个线程。

在当前 KEM 代码中，`KEM_SERIAL_TPB` 控制批量 KEM serial kernel 的启动配置：

```cpp
int tpb = KEM_SERIAL_TPB;
int blocks = (batch_count + tpb - 1) / tpb;

batch_kem_keypair_serial_kernel<<<blocks, tpb>>>(...);
batch_kem_encaps_serial_kernel<<<blocks, tpb>>>(...);
batch_kem_decaps_serial_kernel<<<blocks, tpb>>>(...);
```

以 `batch_count=32768` 为例：

| TPB | Block 数量 | 含义 |
|---:|---:|---|
| 64 | 512 blocks | 每个 block 处理 64 个 KEM 实例 |
| 256 | 128 blocks | 每个 block 处理 256 个 KEM 实例 |
| 512 | 64 blocks | 每个 block 处理 512 个 KEM 实例 |

`TPB` 不是 AMD 特有概念，CUDA/NVIDIA 和 HIP/AMD 都存在类似 kernel launch 配置。区别在于：

> TPB 不是 AMD 特有，但最优 TPB 与 GPU 架构、寄存器压力、occupancy、wavefront/warp 调度、kernel 内部逻辑和访存模式强相关。因此 CUDA 代码迁移到 ROCm 后，不能默认沿用 NVIDIA 上的配置。

当前本地 4090 构建脚本中，`KEM_SERIAL_TPB` 默认值为：

```bash
KEM_SERIAL_TPB="${KEM_SERIAL_TPB:-64}"
```

也就是说，原 4090 版本默认使用：

```text
KEM_SERIAL_TPB=64
```

而 AMD gfx1100 上的 Kyber-768 实测显示：

| TPB | Keygen | Encaps | Decaps |
|---:|---:|---:|---:|
| 32 | 4.91M ops/s | 5.98M ops/s | 5.45M ops/s |
| 64 | 4.87M ops/s | 5.99M ops/s | 5.54M ops/s |
| 128 | 5.08M ops/s | 5.89M ops/s | 5.53M ops/s |
| 256 | 5.44M ops/s | 6.04M ops/s | 5.58M ops/s |
| 512 | 5.05M ops/s | 5.60M ops/s | 5.23M ops/s |

结论：

> 将 Kyber-768 的 `KEM_SERIAL_TPB` 从 4090 默认沿用的 64 调整到 AMD 更适配的 256 后，keygen 吞吐从 4.87M ops/s 提升到 5.44M ops/s，提升约 11.9%。这说明简单 HIP 迁移只能保证“能跑”，要发挥 ROCm/RDNA3 性能，必须重新进行平台相关的 kernel launch 参数调优。

适合论文/PPT 的一句话：

> HIP 迁移解决可运行性，ROCm 原生调优决定能否跑满；TPB sweep 是本项目第一项可量化的 ROCm 平台调优结果。

## 10. 下一阶段：使用 ROCm 工具定位瓶颈

当前已有程序内 profile 显示，Kyber/Aigis-enc keygen 的主要瓶颈集中在 sample 阶段。但程序内 profile 只能给出粗粒度阶段耗时，下一步需要使用 ROCm 工具定位到 kernel 级别。

目标：

1. 确认 sample 阶段对应哪些 kernel 或设备函数路径。
2. 观察 kernel 执行时间、调用次数、grid/block 配置。
3. 判断是否存在 occupancy 不足、寄存器压力、LDS/显存访问效率低、kernel launch 过多等问题。
4. 将 profile 结论转化为下一步代码优化方案。

建议优先分析三个目标：

```text
kyber768_amd      中等安全等级，已有 TPB 正向优化
kyber1024_amd     高安全等级 Kyber，sample 与 matvec 更重
aigisenc4_amd     高安全等级 Aigis-enc，sample 占比最高
```

### 10.1 先确认 ROCm 工具可用性

在 AMD JupyterLab 中执行：

```bash
which rocprofv3 || true
which rocprof || true
which rocm-smi || true
hipcc --version
```

记录：

- ROCm 工具是否存在。
- `rocprofv3` 是否可运行。
- `rocm-smi` 是否能读取显卡信息。
- `hipcc` 版本。

实际结果：

```text
/opt/python/bin/rocprofv3
/opt/python/bin/rocm-smi
HIP version: 7.12.60610-2bd1678d3d
AMD clang version 22.0.0git
Target: x86_64-unknown-linux-gnu
InstalledDir: /opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin
```

结论：

> 当前 AMD JupyterLab 环境提供 `rocprofv3` 和 `rocm-smi`，没有发现旧版 `rocprof`。后续性能定位应以 `rocprofv3 --kernel-trace --hip-trace` 为主。

### 10.2 使用 rocprofv3 做 kernel trace

当前脚本：

```bash
bash profile_kem_one_amd.sh kyber768_amd 32768 3
```

如果 `amd_results/profile/xxx_rocprof/` 目录为空，需要改用更显式的 `rocprofv3` 参数。可尝试：

```bash
mkdir -p amd_results/profile/kyber768_manual_rocprof

rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --output-format csv \
  --output-directory amd_results/profile/kyber768_manual_rocprof \
  -- \
  ./kyber768_amd --batch 32768 --n-ops 1 --no-correctness --pipeline
```

然后查看输出：

```bash
find amd_results/profile/kyber768_manual_rocprof -type f -maxdepth 2 -print
```

如果有 CSV 文件，再压缩或 `cat` 关键文件给后续分析。

实际 `rocprofv3` 运行成功：

```text
rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --output-format csv \
  --output-directory amd_results/profile/kyber768_manual_rocprof \
  -- \
  ./kyber768_amd --batch 32768 --n-ops 1 --no-correctness --pipeline
```

生成文件：

```text
amd_results/profile/kyber768_manual_rocprof/nb-a8b881d6/3984_kernel_trace.csv
amd_results/profile/kyber768_manual_rocprof/nb-a8b881d6/3984_hip_api_trace.csv
amd_results/profile/kyber768_manual_rocprof/nb-a8b881d6/3984_agent_info.csv
```

该结果说明：

> ROCm 工具链已经可以采集 Kyber-768 的 kernel trace 与 HIP API trace。下一步需要解析 `kernel_trace.csv`，找出耗时最高的 kernel，并与程序内 profile 的 sample/NTT/matvec/pack 阶段进行对应。

### 10.3 如果 rocprofv3 不可用，使用 rocprof

有些环境可能只有旧版 `rocprof`。可尝试：

```bash
mkdir -p amd_results/profile/kyber768_rocprof_old

rocprof \
  --hip-trace \
  --hsa-trace \
  -d amd_results/profile/kyber768_rocprof_old \
  ./kyber768_amd --batch 32768 --n-ops 1 --no-correctness --pipeline
```

查看输出：

```bash
find amd_results/profile/kyber768_rocprof_old -type f -maxdepth 2 -print
```

### 10.4 同步采集 GPU 状态

在 benchmark 前后记录：

```bash
rocm-smi
rocm-smi --showuse --showmemuse --showtemp --showpower
```

用于论文中的资源描述：

- GPU 型号与架构。
- 显存容量。
- 运行时显存使用。
- GPU utilization。
- 功耗/温度可选。

### 10.5 下一步可能的优化方向

根据目前数据，优先级如下：

1. **按操作拆分 TPB**
   - 当前单一 `KEM_SERIAL_TPB` 同时影响 keygen/encaps/decaps。
   - 下一步可改成 `KEM_KEYGEN_TPB=256`、`KEM_ENCAPS_TPB=64`、`KEM_DECAPS_TPB=64`。
   - 目标：保留 keygen 提升，同时避免 encaps/decaps 下降。

2. **sample 路径优化**
   - split sample 实验证明简单拆 kernel 会负优化。
   - 后续应考虑保持数据局部性，减少中间全局内存写回，而不是盲目拆分。

3. **按算法选择配置**
   - Kyber 系列明显更受益于 `TPB=256`。
   - Aigis-enc 系列需要单独扫参，不适合直接套用 Kyber 配置。

4. **端到端工作流优化**
   - 小 batch 吞吐较低。
   - 多文件科研数据流转平台应聚合任务，使用批处理摊薄 launch 和同步开销。

## 11. rocprofv3 Kernel Trace 初步结论

`rocprofv3` 已成功生成并解析 Kyber-768 的 kernel trace 与 HIP API trace：

```text
amd_results/profile/kyber768_manual_rocprof/nb-a8b881d6/3984_kernel_trace.csv
amd_results/profile/kyber768_manual_rocprof/nb-a8b881d6/3984_hip_api_trace.csv
```

### 11.1 Kernel 热点

Top kernels by total time：

| Kernel | Total | Avg | Calls | 说明 |
|---|---:|---:|---:|---|
| `batch_kem_decaps_serial_kernel` | 11.728 ms | 5.864 ms | 2 | decaps 主路径 |
| `batch_kem_encaps_serial_kernel` | 9.944 ms | 4.972 ms | 2 | encaps 主路径 |
| `batch_kem_keypair_serial_kernel` | 9.564 ms | 9.564 ms | 1 | serial keygen 主路径 |
| `batch_keygen_warp_sample_kernel` | 5.130 ms | 5.130 ms | 1 | pipeline keygen 采样阶段 |
| `batch_polyvec_matvec_kernel` | 0.485 ms | 0.485 ms | 1 | pipeline matvec |
| `batch_pack_keypair_finalize_kernel` | 0.351 ms | 0.351 ms | 1 | keypair finalize |
| `batch_invntt_kernel` | 0.334 ms | 0.111 ms | 3 | inverse NTT |
| `batch_ntt_kernel` | 0.303 ms | 0.101 ms | 3 | NTT |
| `batch_poly_caddq_kernel` | 0.265 ms | 0.044 ms | 6 | modular normalize |
| `batch_poly_add_kernel` | 0.169 ms | 0.056 ms | 3 | polynomial add |
| `batch_pack_pk_polyvec_kernel` | 0.109 ms | 0.109 ms | 1 | pack pk |
| `batch_pack_sk_polyvec_kernel` | 0.105 ms | 0.105 ms | 1 | pack sk |

注意：

> 本次命令带有 `--pipeline`，程序会先跑 serial 测试，再跑 pipeline 测试。因此 trace 中同时包含 serial keygen/encaps/decaps kernel 和 pipeline keygen 分阶段 kernel。不能直接把所有 kernel 总时间相加作为单一路径耗时，应按执行路径分开解释。

### 11.2 Kernel 级瓶颈判断

对 pipeline keygen 而言，kernel trace 与程序内 profile 一致：

```text
batch_keygen_warp_sample_kernel: 5.130 ms
batch_polyvec_matvec_kernel:    0.485 ms
batch_invntt_kernel total:      0.334 ms
batch_ntt_kernel total:         0.303 ms
pack pk/sk/finalize total:      0.565 ms 左右
```

结论：

> ROCm kernel trace 进一步确认，Kyber-768 pipeline keygen 的主要热点是 `batch_keygen_warp_sample_kernel`，即采样/XOF/拒绝采样路径。NTT、inverse NTT、matvec 和 packing 的耗时均显著低于 sample。因此下一步优化重点应继续围绕采样路径，而不是优先重写 NTT。

### 11.3 HIP API Trace

Top HIP APIs by total time：

| HIP API | Total | Avg | Calls | 说明 |
|---|---:|---:|---:|---|
| `hipGetDevice` | 172.122 ms | 172.122 ms | 1 | 初始化/查询开销 |
| `hipMemcpy` | 89.495 ms | 22.374 ms | 4 | 主机到设备数据准备 |
| `hipDeviceSynchronize` | 39.728 ms | 4.966 ms | 8 | 包含等待 kernel 完成时间 |
| `hipLaunchKernel` | 2.147 ms | 0.086 ms | 25 | kernel launch 开销 |
| `hipFree` | 1.114 ms | 0.062 ms | 18 | 释放显存 |
| `hipMalloc` | 0.452 ms | 0.025 ms | 18 | 分配显存 |

解释：

- `hipGetDevice` 是一次性初始化/查询开销，不应计入稳定态吞吐瓶颈。
- `hipDeviceSynchronize` 的时间包含等待 GPU kernel 执行完成，不代表纯 CPU API 开销。
- `hipMemcpy` 时间较高，说明端到端工作流中需要避免频繁 host-device 拷贝。
- `hipLaunchKernel` 总耗时约 2.147 ms，25 次调用，平均约 0.086 ms；对大 batch 影响可接受，但对小 batch 会明显影响端到端效率。

### 11.4 从 ROCm Trace 得出的优化建议

1. **sample kernel 是第一优化目标**
   - 重点分析 `batch_keygen_warp_sample_kernel` 中 SHAKE/XOF、拒绝采样、噪声采样的线程协作方式。
   - split sample 已经证明会负优化，因此后续应优先考虑保持数据局部性与减少全局内存中间写回。

2. **减少端到端内存搬运**
   - `hipMemcpy` 在 API trace 中占比较高。
   - 后续多文件可信流转平台应尽量复用 device buffer，避免每批数据重复分配和拷贝。

3. **减少小 batch 下 launch/sync 开销**
   - `hipLaunchKernel` 与 `hipDeviceSynchronize` 对小 batch 会放大。
   - 前端/工作流层应聚合多文件任务，使用 batch 队列提高 GPU 利用率。

4. **按路径分别优化**
   - serial keygen/encaps/decaps 是当前 best-throughput 基线。
   - pipeline keygen 适合用于分阶段 profile，但不一定是所有参数集的性能最优实现。

### 11.5 最终配置下的资源分析

使用最终候选配置重新采集：

```text
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
batch=32768
rocprofv3 --kernel-trace --hip-trace
```

关键 kernel 资源摘要：

| Kernel | Total | Calls | VGPR | SGPR | Scratch | LDS | Workgroup | Grid |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `batch_kem_decaps_serial_kernel` | 11.569 ms | 2 | 184 | 128 | 17168 | 0 | 128x1x1 | 32768x1x1 |
| `batch_kem_encaps_serial_kernel` | 9.957 ms | 2 | 184 | 128 | 16064 | 0 | 128x1x1 | 32768x1x1 |
| `batch_kem_keypair_serial_kernel` | 8.844 ms | 1 | 184 | 128 | 8592 | 0 | 256x1x1 | 32768x1x1 |
| `batch_keygen_warp_sample_kernel` | 5.101 ms | 1 | 152 | 128 | 304 | 512 | 128x1x1 | 1048576x1x1 |
| `batch_polyvec_matvec_kernel` | 0.477 ms | 1 | 16 | 128 | 0 | 0 | 256x1x1 | 8388608x3x1 |
| `batch_invntt_kernel` | 0.329 ms | 3 | 16 | 128 | 0 | 1024 | 128x1x1 | 4194304x1x1 |
| `batch_ntt_kernel` | 0.304 ms | 3 | 16 | 128 | 0 | 1024 | 128x1x1 | 4194304x1x1 |

资源侧结论：

> ROCm kernel trace 显示，monolithic serial KEM kernel 的 VGPR 均达到 184，且 encaps/decaps scratch 分别达到 16064/17168 bytes，说明当前单线程单实例的设备函数路径存在较高寄存器与栈/溢出压力。相比之下，NTT、inverse NTT 和 matvec 的 VGPR 较低且 scratch 为 0，说明它们不是当前资源瓶颈。

由此得到两个优化方向：

1. **sample/XOF/rejection sampling 仍是算法阶段热点**
   - `batch_keygen_warp_sample_kernel` 耗时 5.101 ms，仍是 pipeline keygen 中最大单项。
   - 该 kernel 使用 VGPR=152、Scratch=304、LDS=512，说明 sample 路径也存在较高寄存器压力，但比 monolithic serial kernel 的 scratch 小得多。

2. **monolithic serial kernel 存在寄存器/栈压力**
   - keypair/encaps/decaps serial kernel 均使用 VGPR=184。
   - encaps/decaps scratch 超过 16KB，后续可考虑拆出局部大数组、减少设备函数局部状态、复用全局/共享缓冲，或针对 encaps/decaps 建立更细粒度 batch pipeline。

HIP API trace：

| API | Total | Calls | 判断 |
|---|---:|---:|---|
| `hipGetDevice` | 173.320 ms | 1 | 初始化/查询开销，不计入稳定态瓶颈 |
| `hipMemcpy` | 87.502 ms | 4 | 数据准备开销，端到端系统应减少 host-device 拷贝 |
| `hipDeviceSynchronize` | 38.816 ms | 8 | 包含等待 GPU kernel 完成，不是纯 API 开销 |
| `hipLaunchKernel` | 2.002 ms | 25 | 小 batch 场景会放大 launch 开销 |
| `hipMalloc/hipFree` | 1.542 ms | 36 | 后续工作流应复用 device buffer |

工程化建议：

> 对论文而言，当前 ROCm trace 可以支撑“sample 是算法热点，monolithic KEM kernel 有寄存器/栈压力，端到端系统需减少拷贝和复用显存缓冲”三个结论。下一步调优不应优先优化 NTT，而应优先围绕 sample 路径、serial kernel 局部状态和端到端缓冲复用展开。

### 11.6 rocm-smi 资源采集说明

单次运行后执行：

```bash
rocm-smi --showuse --showmemuse --showtemp --showpower
```

得到 GPU 利用率与显存占用均为 0。这是因为命令在 benchmark 结束后执行，GPU 已经空闲，不能代表运行中资源占用。

当前可记录的静态信息：

- 服务器暴露 8 张 AMD GPU。
- 空闲温度约 24 到 33 摄氏度。
- 空闲功耗约 8 到 14 W。
- benchmark 结束后 VRAM 使用率恢复为 0。

后续如果要获得运行中资源数据，需要在 benchmark 期间循环采样 `rocm-smi`。

### 11.7 运行中 rocm-smi 采样结果

使用长迭代 benchmark 期间循环采样：

```bash
(
  for i in $(seq 1 80); do
    echo "===== sample $i $(date '+%H:%M:%S.%3N') ====="
    rocm-smi --showuse --showmemuse --showtemp --showpower
    sleep 0.2
  done
) > amd_results/resource/rocm_smi_during_kyber768.log &

./kyber768_amd --batch 32768 --n-ops 200 --no-correctness
wait
```

性能结果：

```text
--- batch=32768 n_ops=200 mode=serial ---
  Keygen:      4.9 ms/batch -> 6637753 ops/sec
  Encaps:      5.5 ms/batch -> 6008051 ops/sec
  Decaps:      6.0 ms/batch -> 5440166 ops/sec
```

运行中资源观测：

| 指标 | 观测值 |
|---|---|
| GPU use | 峰值 100%，多次采样 99%-100% |
| VRAM allocated | 峰值约 8% |
| Average graphics package power | 约 237-243 W |
| Edge temperature | 约 28-35 C |
| Junction temperature | 约 36-46 C |
| Memory temperature | 约 30-38 C |

资源侧结论：

> 长迭代 Kyber-768 benchmark 期间 GPU use 可稳定达到 99%-100%，说明当前批处理工作负载能够充分占用 AMD GPU 计算资源。与此同时，VRAM 使用峰值仅约 8%，说明当前 KEM benchmark 并非显存容量受限，更可能受采样/XOF、寄存器压力、scratch/栈压力和单 kernel 计算路径影响。

适合论文表述：

> `rocm-smi` 运行中采样显示，Kyber-768 批处理阶段 GPU 利用率接近 100%，显存占用约 8%，功耗约 240W。结合 `rocprofv3` 的 kernel trace，可判断当前瓶颈主要来自计算密集型采样与 monolithic kernel 资源压力，而非显存容量不足。

### 11.8 launch_bounds 关闭实验

实验配置：

```bash
KEM_KEYPAIR_LAUNCH_BOUNDS=0
KEM_ENCAPS_LAUNCH_BOUNDS=0
KEM_DECAPS_LAUNCH_BOUNDS=0
KEM_KEYGEN_TPB=256
KEM_ENCAPS_TPB=128
KEM_DECAPS_TPB=128
```

测试：

```text
--- batch=32768 n_ops=50 mode=serial ---
  Keygen:      5.0 ms/batch -> 6563939 ops/sec
  Encaps:      5.5 ms/batch -> 5997218 ops/sec
  Decaps:      5.9 ms/batch -> 5507774 ops/sec
```

结论：

> 关闭 `__launch_bounds__` 后吞吐与 `n_ops=200` 稳定结果基本接近，没有出现明显提升。因此当前性能瓶颈不主要来自 launch bounds 约束本身。后续调优应转向 sample 内部实现、设备函数局部状态、缓冲复用和端到端数据搬运优化。

## 15. 端到端工程优化：Device Buffer 复用

为模拟后续“多文件科研数据可信流转平台”中的连续批处理场景，新增 `--reuse-bench` 测试入口，对比两种端到端执行方式：

1. **Alloc-each-round**
   - 每轮重新 `cudaMalloc`
   - 拷贝输入 seed
   - 执行 keygen/encaps/decaps
   - 每轮 `cudaFree`

2. **Reuse buffers**
   - 初始化时分配一次 device buffer
   - 每轮复用已有 buffer
   - 只更新输入 seed 并执行 kernel

测试命令：

```bash
KEM_KEYGEN_TPB=256 KEM_ENCAPS_TPB=128 KEM_DECAPS_TPB=128 \
bash build_hip.sh kyber768

./kyber768_amd --batch 32768 --n-ops 5 --reuse-bench 20 --no-correctness
```

结果：

```text
=== Buffer reuse benchmark: Kyber-768 ===
batch=32768 rounds=20 n_ops_per_round=5
  Alloc-each-round: total=  1701.7 ms | per_round= 85.085 ms | full-kem throughput=1925608 instances/sec
  Reuse buffers:     total=  1599.8 ms | per_round= 79.991 ms | full-kem throughput=2048239 instances/sec
  Reuse speedup:     1.064x
```

结论：

> Device buffer 复用使 Kyber-768 端到端 full-KEM 批处理吞吐从 1.93M instances/s 提升到 2.05M instances/s，提升约 6.4%。该优化不改变单个密码 kernel 内部算法，而是减少多批次工作流中的显存分配/释放开销，更符合真实科研数据平台的连续处理模式。

工程意义：

> 在真实多文件可信流转系统中，服务端会连续处理多个文件批次。为每批文件重新分配和释放 GPU 缓冲会引入额外运行时开销。通过维护长期复用的 device buffer pool，可以提升端到端吞吐，并与前端/后端任务队列自然结合。

适合论文/PPT 表述：

> 除 kernel 级 TPB 调优外，本项目还针对真实工作流进行端到端优化。通过复用 GPU device buffer，Kyber-768 full-KEM 连续批处理吞吐提升 6.4%，说明 ROCm 平台性能优化不仅包括 kernel 内部调参，也包括面向应用工作流的内存管理与任务调度优化。



### Step 2：全量构建 7 个 KEM 目标

目的：

确认不只是 Kyber-768，Kyber512/1024 与 Aigis-enc 1/2/3/4 都能在 AMD 上构建和冒烟。

命令：

```bash
cd /app/kyberandaigis-enc
bash build_hip.sh
bash run_kem_smoke_amd.sh
cat amd_results/kem_smoke_summary.csv
```

需要关注：

- 哪些目标可以编译。
- 哪些目标正确性 PASS。
- 哪些目标出现 stack、寄存器、显存或运行时错误。

### Step 3：全量 batch sweep

目的：

找到各算法的最佳 batch size，并观察 AMD 上吞吐峰值。

命令：

```bash
bash run_kem_sweep_amd.sh
cat amd_results/kem_best.csv
```

需要输出：

- 每个算法 keygen/encaps/decaps 最佳吞吐。
- 每个算法最佳 batch 配置。
- 与 4090 数据对比。

### Step 4：针对热点目标做 profile

优先目标：

```text
kyber768_amd
kyber1024_amd
aigisenc3_amd
aigisenc4_amd
```

命令示例：

```bash
bash profile_kem_one_amd.sh kyber1024_amd 32768 3
bash profile_kem_one_amd.sh aigisenc4_amd 32768 3
```

需要分析：

- 是否仍然是 sample 阶段主导。
- 大参数集是否出现 pack、NTT、matvec 或 decaps 瓶颈。
- Aigis-enc 与 Kyber 的瓶颈是否一致。

### Step 5：将 KEM 结果接入项目主线

短期目标：

- 先形成 `Kyber/Aigis-enc AMD 跑通 + 性能 + 瓶颈分析` 小节。
- 再与已有 ML-DSA/Aigis-sig AMD 结果合并。
- 最后落地到“多文件科研数据可信流转平台”。

项目主线表述：

> 本项目不是单一 Kyber 跑分，而是面向 AMD ROCm 生态构建后量子科研数据可信流转系统。Kyber/Aigis-enc 负责会话密钥封装，ML-DSA/Aigis-sig 负责数据签名与篡改检测。通过 ROCm 上的批处理、profile 和优化，验证 AMD GPU 在后量子密码高并发科研数据流转中的工程价值，并为 AMD 构建类 cuPQC 的 ROCm-PQC 库提供实验依据。

## 8. 下一步优先级

最高优先级：

1. 跑 `kyber768_split_amd`，验证 sample 拆分是否能降低 70% 瓶颈。
2. 全量 `bash build_hip.sh`，确认 7 个 KEM 目标的编译情况。
3. 全量 `bash run_kem_smoke_amd.sh`，确认正确性矩阵。
4. 全量 `bash run_kem_sweep_amd.sh`，拿到 AMD KEM 最佳性能表。
5. 选 2 个代表目标做 profile，形成可写入论文的瓶颈定位表。

当前最值得追的优化点：

> Kyber-768 keygen sample 阶段占 70%，优先围绕采样/XOF/拒绝采样并行化做 ROCm 调优。
