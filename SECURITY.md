# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository. For vulnerabilities in third-party dependencies, report upstream. For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).

## Automated security scanning

Alongside the reporting path above, this repository is scanned automatically.
The scanners are not implemented here: Quartz calls the shared
[`ROCm/rocm-security-gh`](https://github.com/ROCm/rocm-security-gh)
`security-baseline.yml` reusable workflow, so the scanner versions and
behavior are maintained centrally for ROCm, and this repository supplies only
its own configuration (the `*.yml` / `*.toml` files at the repo root).

| Scanner                                          | Looks for                                                | Configuration                        |
| ------------------------------------------------ | -------------------------------------------------------- | ------------------------------------ |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secrets and credentials in tracked files and git history | [`gitleaks.toml`](gitleaks.toml)     |
| [bandit](https://bandit.readthedocs.io/)         | Unsafe patterns in our Python scripts                    | [`bandit.yml`](bandit.yml)           |
| [zizmor](https://docs.zizmor.sh/)                | GitHub Actions workflow vulnerabilities                  | [`zizmor.yml`](zizmor.yml)           |
| [trivy](https://trivy.dev/)                      | Dependency vulnerabilities and misconfigurations         | [`trivy.yml`](trivy.yml)             |
| [CodeQL](https://codeql.github.com/)             | Semantic code analysis of our Python                     | org-wide default (no local override) |

Two workflows run them, and where a finding shows up depends on which one
produced it:

- [`security_scan_pr.yml`](.github/workflows/security_scan_pr.yml) runs on
  every pull request, scoped to what that pull request changed. It reports in
  the job summary and a build artifact, and deliberately does not upload to
  the Security tab, so pull requests from forks behave the same as those from
  branches in this repository.
- [`security_scan_weekly.yml`](.github/workflows/security_scan_weekly.yml)
  runs on a schedule across the whole repository and uploads SARIF to this
  repository's Security tab, which is the authoritative view of the current
  state. Quartz is a monorepo-adjacent, low-churn repository, so a weekly
  cadence (rather than on every push to `develop`/`main`) is enough to keep
  the Security tab current without adding a scan to every merge.

> [!IMPORTANT]
> A finding from these scanners is not a vulnerability report. If a scanner
> finding turns out to be an exploitable vulnerability in shipped ROCm
> software, report it through the AMD Product Security portal above rather
> than in a public issue or pull request.

Contributors can run every scanner locally against the same configuration CI
uses; see
[the security scanners section in `CONTRIBUTING.md`](CONTRIBUTING.md#security-scanners).
