# table_sig_local_winners_feature_matrix_2026-06-16

## Local winners from feature matrix

These rows are evidence for workload-sensitive tuning. They are not final default policies.

| Target | Mode | Batch | Local winner | Speedup vs base | Sign ops/s |
| --- | --- | ---: | --- | ---: | ---: |
| Aigis-sig1 | independent | 8192 | cp_fuse | 1.2375 | 64504 |
| Aigis-sig1 | independent | 16384 | wave64_ctrl | 1.2076 | 86532 |
| Aigis-sig1 | independent | 32768 | wave64_ctrl | 1.0720 | 71926 |
| Aigis-sig1 | paper | 1024 | wave64_ctrl | 1.4385 | 21375 |
| Aigis-sig1 | paper | 8192 | adaptive | 1.0853 | 63246 |
| Aigis-sig1 | paper | 16384 | wave64_ctrl | 1.2519 | 82738 |
| Aigis-sig1 | paper | 32768 | tail16_base | 1.0767 | 72992 |
| Aigis-sig2 | independent | 1024 | yhat_dup | 1.2031 | 12854 |
| Aigis-sig2 | independent | 8192 | wave64_ctrl | 1.1985 | 41570 |
| Aigis-sig2 | independent | 16384 | adaptive | 1.1586 | 50797 |
| Aigis-sig2 | paper | 1024 | check8 | 1.3242 | 14708 |
| Aigis-sig2 | paper | 8192 | cp_fuse | 1.2171 | 38465 |
| Aigis-sig2 | paper | 16384 | wave64_ctrl | 1.0752 | 50137 |
| Aigis-sig2 | paper | 32768 | adaptive | 1.0998 | 49197 |
| Aigis-sig3 | independent | 1024 | adaptive | 1.1214 | 12032 |
| Aigis-sig3 | independent | 16384 | wave64_ctrl | 1.4200 | 42911 |
| Aigis-sig3 | independent | 32768 | cp_fuse | 1.1429 | 39925 |
| Aigis-sig3 | paper | 8192 | adaptive | 1.2261 | 37768 |
| Aigis-sig3 | paper | 16384 | adaptive | 1.0331 | 41904 |
| ML-DSA-44 | independent | 1024 | cp_fuse | 1.3923 | 39625 |
| ML-DSA-44 | independent | 8192 | wave64_ctrl | 1.0026 | 99262 |
| ML-DSA-44 | independent | 16384 | wave64_ctrl | 1.0956 | 98699 |
| ML-DSA-44 | independent | 32768 | tail16_base | 1.1110 | 100649 |
| ML-DSA-44 | paper | 1024 | cp_fuse | 1.3053 | 39605 |
| ML-DSA-44 | paper | 16384 | tail16_cp_fuse | 1.0543 | 102919 |
| ML-DSA-44 | paper | 32768 | tail16_base | 1.1936 | 98810 |
| ML-DSA-65 | independent | 1024 | check8 | 1.0973 | 24036 |
| ML-DSA-65 | independent | 8192 | check16 | 1.2116 | 55130 |
| ML-DSA-65 | independent | 16384 | cp_fuse | 1.3625 | 62591 |
| ML-DSA-65 | independent | 32768 | wave64_ctrl | 1.2118 | 60904 |
| ML-DSA-65 | paper | 1024 | yhat_dup | 1.0289 | 19158 |
| ML-DSA-65 | paper | 16384 | tail16_base | 1.1451 | 61008 |
| ML-DSA-65 | paper | 32768 | tail16_base | 1.1380 | 56954 |
| ML-DSA-87 | independent | 1024 | tail16 | 1.5159 | 22648 |
| ML-DSA-87 | independent | 8192 | cp_fuse | 1.0052 | 50385 |
| ML-DSA-87 | independent | 32768 | cp_fuse | 1.0854 | 48998 |
| ML-DSA-87 | paper | 1024 | yhat_dup | 1.1197 | 16756 |
| ML-DSA-87 | paper | 16384 | tail16_cp_fuse | 1.1680 | 46352 |
| ML-DSA-87 | paper | 32768 | tail16_cp_fuse | 1.1303 | 47494 |

## Interpretation

The table shows that optimization is workload-sensitive. Large local speedups exist, but they do not satisfy the conservative no-regression rule across all measured cells. Therefore, local winners should be discussed as candidate evidence instead of being merged into the stable default build.
