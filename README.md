# HAMT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24165455.svg)](https://doi.org/10.5281/zenodo.24165455)

HAMT is a hygrothermal (Heat, Air and Moisture Transport) calculation library written in Rust.
It simulates coupled, transient heat and moisture transport through multi-layer building
constructions and ships with a drop-in TRNSYS component (Type 395).

## Components

| Crate | Description |
|:---|:---|
| [`hamt-core`](hamt-core) | Solver library: material and construction model, cell grid, boundary conditions, time stepping. |
| [`trnsys-hamt-rs`](trnsys-hamt-rs) | TRNSYS Type 395 DLL built on `hamt-core` for complex simulation in TRNSYS. Tested with [better-trnlizard](https://github.com/JunfXiao/better-trnlizard) in TRNSYS 18. |

> **Planned:** a Python package (PyO3 bindings for `hamt-core`, published on PyPI).

## Features

- Coupled heat and moisture balance with moisture-dependent properties: sorption isotherm,
  liquid suction and redistribution, vapour diffusion and moisture-dependent thermal conductivity.
- Implicit finite-volume discretisation with a block-tridiagonal solver (1-D) and an ADI
  scheme for 2-D/3-D grids, plus adaptive sub-stepping within each macro timestep.
- Material and construction input in the EnergyPlus HAMT format
  (`MaterialProperty:HeatAndMoistureTransfer:*`, `Construction`, `SurfaceProperties:VaporCoefficients`).
- Interior and exterior films (convection, long-wave radiation, vapour transfer) and a
  user-defined node at a given depth.

## Validation

### DIN EN 15026 Annex B
The test suite runs the DIN EN 15026 (Annex B) benchmark and compares temperature and water
content profiles after 7, 30 and 365 days with the reference solution
(|ΔT| ≤ 0.3 K, relative water-content deviation ≤ 2.5 %).
Each release contains the full comparison (`din_en_15026_benchmark.csv`) and a summary in the
release notes. 

It's extremely fast: the whole benchmark test file (365 simulated days) finishes in about 2 s,
even on a shared GitHub-hosted runner.

## Building

Requires the Rust nightly toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml)
(`rustup` installs it automatically).

```bash
cargo test -p hamt-core --release
cargo build --release -p hamt-core
```

The Type 395 DLL links against `TRNDll64.lib` from your own TRNSYS installation
(`TRNSYS18/Exe/`). This file is part of TRNSYS and is **not** distributed with this repository;
copy it to `trnsys-hamt-rs/lib/` before building:

```bash
cargo build --release -p trnsys_hamt_rs
```

Pre-built `trnsys_hamt_rs.dll` files are attached to the
[GitHub releases](https://github.com/JunfXiao/hamt/releases).

## Contributing

Issues and pull requests are welcome!

- **Bug reports / questions:** please [open an issue](https://github.com/JunfXiao/hamt/issues)
  and, if possible, attach the material/construction file, the TRNSYS version and the
  `type_395_*.log` files.
- **Pull requests:** please make sure `cargo test -p hamt-core --release` passes (including the
  DIN EN 15026 benchmark) and never commit `TRNDll64.lib` or other TRNSYS files.

## Citation

If you use HAMT in your work, please cite it via its Zenodo DOI
[10.5281/zenodo.24165455](https://doi.org/10.5281/zenodo.24165455) (all versions), or use
"Cite this repository" on right section of the page.

## License

Licensed under the [Apache License 2.0](LICENSE).
Developed by [Junfeng Xiao](mailto:junfeng.xiao@tum.de) at the Technical University of Munich, Germany.
