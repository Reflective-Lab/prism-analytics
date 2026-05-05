# Contributing to prism

prism is a Converge extension. Contributions follow the same conventions as the Converge foundation.

## Development

```sh
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

While the Converge platform is unreleased, this workspace patches the relevant crates to local checkouts via `[patch.crates-io]`. You need both repos checked out side by side:

```
~/dev/
├── work/converge/
└── extensions/prism/
```

## Boundaries

prism implements analytics **Suggestors** (purposeful: feature extraction, training, evaluation, monitoring) on top of analytics **Backends** (operational: Burn inference, Polars feature pipelines). See the foundation's [Plug Boundary](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/Plug%20Boundary.md) for the layering rule.

## No `unsafe`

The workspace forbids `unsafe`.

## Pull Requests

- Keep PRs small and focused.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Run `cargo check --workspace` before pushing.

## License

By contributing, you agree your contributions are licensed under MIT.
