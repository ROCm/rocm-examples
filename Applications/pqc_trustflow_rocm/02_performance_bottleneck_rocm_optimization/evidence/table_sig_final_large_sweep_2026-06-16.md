# table_sig_final_large_sweep_2026-06-16

## Final large sweep summary

All targets use the stable `base` build for the final selected path.

| Family | Target | Mode | Keygen batch | Keygen ms | Keygen ops/s | Sign batch | Sign ms | Sign ops/s | Verify batch | Verify ms | Verify ops/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ML-DSA | 44 | independent | 32768 | 34.328 | 954556 | 32768 | 299.174 | 109528 | 8192 | 1.211 | 6763774 |
| ML-DSA | 44 | paper | 16384 | 8.459 | 1936824 | 16384 | 180.552 | 90744 | 8192 | 1.213 | 6753549 |
| ML-DSA | 65 | independent | 32768 | 53.505 | 612429 | 16384 | 257.458 | 63638 | 8192 | 1.734 | 4724442 |
| ML-DSA | 65 | paper | 16384 | 10.908 | 1502057 | 16384 | 262.670 | 62375 | 8192 | 1.725 | 4749933 |
| ML-DSA | 87 | independent | 32768 | 73.646 | 444940 | 32768 | 599.155 | 54690 | 8192 | 2.422 | 3382030 |
| ML-DSA | 87 | paper | 32768 | 28.239 | 1160371 | 16384 | 356.069 | 46014 | 8192 | 2.454 | 3337606 |
| Aigis-sig | 1 | independent | 32768 | 32.023 | 1023273 | 32768 | 425.459 | 77018 | 16384 | 2.105 | 7781688 |
| Aigis-sig | 1 | paper | 32768 | 16.474 | 1989031 | 32768 | 486.455 | 67361 | 16384 | 2.120 | 7728283 |
| Aigis-sig | 2 | independent | 32768 | 42.804 | 765542 | 16384 | 422.524 | 38776 | 8192 | 1.285 | 6375467 |
| Aigis-sig | 2 | paper | 16384 | 9.460 | 1731964 | 16384 | 346.621 | 47268 | 8192 | 1.298 | 6309734 |
| Aigis-sig | 3 | independent | 32768 | 55.544 | 589950 | 16384 | 391.438 | 41856 | 8192 | 1.648 | 4970454 |
| Aigis-sig | 3 | paper | 32768 | 22.033 | 1487207 | 32768 | 798.910 | 41016 | 8192 | 1.668 | 4911425 |

## Notes

- Stable baseline: `decomp-pipeline=on`
- Candidate variants were not promoted to the final build
- This table is the final paper-facing large-batch throughput summary
