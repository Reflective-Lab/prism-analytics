# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report through [GitHub Security Advisories](https://github.com/Reflective-Lab/prism-analytics/security/advisories/new) or by emailing **Kenneth Pernyer** at [kenneth@reflective.se](mailto:kenneth@reflective.se).

You should receive a response within 48 hours.

Please include:

- The version of prism you're using
- A description of the vulnerability
- Steps to reproduce
- Any relevant logs or error messages

## Built-in Security Practices

- `unsafe_code = "forbid"` across the workspace
- Training data and feature stores are caller-provided — no implicit data egress
- Model artifacts are not signed by default; deployers must verify provenance
- Analytics suggestors only propose facts; promotion is governed by the Converge engine

## Shared Responsibility

prism provides analytics and ML capabilities, not a hardened production deployment. Operators are responsible for:

- Encrypting model artifacts and training data at rest
- Restricting network exposure of training/inference endpoints
- Vetting third-party datasets for poisoning or PII
- Backups and retention policy
