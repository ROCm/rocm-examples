# 2026-06-16 签名六参数全优化最终决策

- 日期: 2026-06-16
- 工程: `/app/pqc_rocm_full_20260614/amd_sig_anchor_results_20260605_031411`
- 覆盖目标: `mldsa44 / mldsa65 / mldsa87 / aigis1 / aigis2 / aigis3`
- 覆盖模式: `paper / independent`
- 覆盖批大小: `1024 / 8192 / 16384 / 32768`
- 覆盖候选: `base / adaptive / check8 / check16 / wave64_ctrl / cp_fuse / tail16_base / tail16_cp_fuse / yhat_dup`
- repeat: 2
- 最终结论: 六个目标的稳定 selected build 全部选择 `base`

## Selected Build

保守规则:

```text
non-base variant must:
1. pass every measured cell
2. keep min speedup >= 1.0000
3. reach geomean >= 1.0300
```

结果:

| target | selected variant | reason |
| --- | --- | --- |
| mldsa44 | base | no conservative non-base winner |
| mldsa65 | base | no conservative non-base winner |
| mldsa87 | base | no conservative non-base winner |
| aigis1 | base | no conservative non-base winner |
| aigis2 | base | no conservative non-base winner |
| aigis3 | base | no conservative non-base winner |

对应 `sig_amd_variant_plan.env`:

```bash
SIG_AMD_VARIANT_MLDSA44=base
SIG_AMD_VARIANT_MLDSA65=base
SIG_AMD_VARIANT_MLDSA87=base
SIG_AMD_VARIANT_AIGIS1=base
SIG_AMD_VARIANT_AIGIS2=base
SIG_AMD_VARIANT_AIGIS3=base
```

## 关键判断

全量 feature matrix 说明:

1. 局部 winner 很多。
2. 但没有任何候选能在单个 target 的 8 个组合中全部不退化。
3. 因此最终稳定 build 不能简单选择某个全局宏。
4. 候选优化适合写成 feature matrix / resource-aware policy evidence，而不是稳定默认路径。

## 候选诊断摘要

| target | strongest candidate | geomean | min speedup | wins/losses | decision |
| --- | --- | ---: | ---: | --- | --- |
| mldsa44 | wave64_ctrl | 1.0861 | 0.9924 | 7 / 1 | min below 1.0, keep as candidate |
| mldsa65 | wave64_ctrl | 1.0524 | 0.8368 | 4 / 4 | unstable |
| mldsa87 | cp_fuse | 1.0257 | 0.8743 | 5 / 3 | unstable |
| aigis1 | wave64_ctrl | 1.1217 | 0.9089 | 6 / 2 | strong local candidate, not global |
| aigis2 | adaptive | 1.0680 | 0.9066 | 6 / 2 | strong local candidate, not global |
| aigis3 | wave64_ctrl | 1.0025 | 0.8504 | 3 / 5 | weak/unstable |

## 论文表述

建议写成:

> We keep the resource-aware decomposed pipeline as the stable ROCm baseline. Although feature-matrix candidates such as `wave64_ctrl`, `cp_fuse`, `check8/check16`, `tail16`, and `adaptive` produce local wins, none satisfies the conservative target-level rule across all measured mode/batch cells. We therefore separate stable baseline results from candidate optimization evidence.

中文表达:

> 全量 feature matrix 显示，AMD ROCm 上的签名优化不是单一宏开关可以全局解决的问题。不同 target、benchmark mode 和 batch size 下最优策略不同，且候选策略存在明显回退风险。因此最终稳定构建保留 resource-aware decomp pipeline，候选优化作为局部收益和资源感知调优证据单独报告。

## 下一步

1. 构建 selected build，即 base 六目标。
2. 运行 policy smoke。
3. 运行 debug matrix。
4. 运行 large sweep 生成最终签名性能表。
5. 对代表目标做资源归因 profile。
