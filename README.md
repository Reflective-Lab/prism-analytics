# prism

Analytics and ML pipeline for Converge agents. Implements feature extraction,
inference, training, and monitoring suggestors on top of Polars and Burn.

`prism` is a Converge **extension** — it depends on Converge's stable
contracts (`converge-pack`, `converge-domain`, `converge-optimization`,
`converge-provider`) and lives outside the Converge foundation
repository. See the foundation's
[Plug Boundary](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/Plug%20Boundary.md)
for why.

## Layout

- `crates/prism` — analytics suggestors, ML packs, training pipeline, feature
  engineering, model registry, monitoring, and inference.

## Status

Extracted from `converge/crates/analytics` on 2026-05-05 as part of the v3.8
foundation extraction (ADR-008). Pre-1.0 — no published versions yet.

## Build

```sh
cargo check
cargo build --release
```

While Converge platform crates are unreleased, this workspace patches them to
local checkouts at `../../work/converge/crates/...` via `[patch.crates-io]`.

## License

MIT — see [LICENSE](LICENSE).
