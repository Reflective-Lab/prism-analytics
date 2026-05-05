# prism Agent Guide

This is the canonical agent entrypoint for `prism`.

`prism` is a Converge extension for analytics, ML, feature extraction,
inference, training, monitoring, and analytic packs.

## Start Here

1. Read `README.md`.
2. Read `/Users/kpernyer/dev/extensions/kb/Modules/Prism.md`.
3. Check `Cargo.toml` feature flags for storage and Excel ingestion.
4. Use `just --list` for local commands.

## Commands

```bash
just check
just check-all
just test
just lint
just doc
```

## Boundaries

- Converge owns the pack and suggestor contracts.
- `prism` owns analytic pack implementations and ML pipeline suggestors.
- Products own domain-specific datasets, model rollout decisions, and runtime
  assembly.

## Rules

- Preserve `unsafe_code = "forbid"`.
- Keep pack outputs as proposals, not facts.
- Maintain compile-fail tests for authority boundaries.
- Update `README.md`, `CHANGELOG.md`, and the extensions KB when packs or
  public agents change.
