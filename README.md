# crfs-rs

[![CI](https://github.com/encrystudio/crfs-mod/workflows/CI/badge.svg)](https://github.com/encrystudio/crfs-mod/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/encrystudio/crfs-mod/branch/main/graph/badge.svg)](https://codecov.io/gh/encrystudio/crfs-mod)
[![Crates.io](https://img.shields.io/crates/v/crfs-mod.svg)](https://crates.io/crates/crfs-mod)
[![docs.rs](https://docs.rs/crfs-mod/badge.svg)](https://docs.rs/crfs-mod/)

Modified pure Rust port of CRFsuite: a fast implementation of Conditional Random Fields (CRFs)

This is a fork of [messense/crfs-rs](https://github.com/messense/crfs-rs) with some modifications/additions I personally find useful:

## Modifications

Currently this fork works as a drop-in replacement for the original `crfs` crate. Just change the imports from `crfs` to `crfs_mod`.

### Crate

- Updated dependencies
- Changed name from `crfs` to `crfs-mod` (`crfs_mod` in code)

### Repository

- Removed all python stuff
- Optimized CI workflow
- Removed dependabot
- Removed funding
- Enhanced .gitignore

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
crfs-mod = { git = "https://github.com/encrystudio/crfs-mod", branch = "main" }
```

## Performance

Performance comparsion with CRFsuite on MacBook Pro (13-inch, M4 MAX, 2024)

```bash
$ cargo bench --bench crf_bench -- --output-format bencher
test tag/crfs ... bench:         579 ns/iter (+/- 15)
test tag/crfsuite ... bench:        1185 ns/iter (+/- 20)
```

Last updated on 2026-02-01.

## License

This work is released under the MIT license. A copy of the license is provided
in the [LICENSE](./LICENSE) file.
