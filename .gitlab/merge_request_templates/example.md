## Notes for the reviewer
_The reviewer should acknowledge all these topics._
<insert notes>

## Checklist before merge
- [ ] CMake support is added
    - [ ] Dependencies are copied via `IMPORTED_RUNTIME_ARTIFACTS` if applicable
- [ ] GNU Make support is added (Linux)
- [ ] Visual Studio project is added for VS2017, 2019, 2022 (Windows) (use [the script](https://projects.streamhpc.com/departments/knowledge/employee-handbook/-/wikis/Projects/AMD/Libraries/examples/Adding-Visual-Studio-Projects-to-new-examples#scripts))
    - [ ] DLL dependencies are copied via `<Content Include`
    - [ ] Visual Studio project is added to `ROCm-Examples-vs*.sln` (ROCm)
    - [ ] Visual Studio project is added to `ROCm-Examples-Portable-vs*.sln` (ROCm/CUDA) if applicable
- [ ] Inline code documentation is added
- [ ] README is added according to template
    - [ ] Related READMEs, ToC are updated
- [ ] The CI passes for Linux/ROCm, Linux/CUDA, Windows/ROCm, Windows/CUDA.
