## Notes for the reviewer

_The reviewer should acknowledge all these topics._
<insert notes>

## Checklist before merge

- [ ] CMake support is added
  - [ ] Dependencies are copied via `IMPORTED_RUNTIME_ARTIFACTS` if applicable
- [ ] GNU Make support is added (Linux)
- [ ] Visual Studio project definitions are added to the [Examples Registry](https://projects.streamhpc.com/amd/libraries/examples-registry). See [Examples Registry - adding a new example](https://projects.streamhpc.com/amd/libraries/examples-registry#adding-a-new-example) and [Examples Registry - registry file documentation](https://projects.streamhpc.com/amd/libraries/examples-registry#registry).
- [ ] Inline code documentation is added
- [ ] README is added according to template
  - [ ] Related READMEs, ToC are updated
- [ ] The CI passes for Linux/ROCm, Linux/CUDA, Windows/ROCm, Windows/CUDA.
