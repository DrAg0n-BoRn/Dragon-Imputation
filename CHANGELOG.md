# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [0.2.1] - 2026-10-14

### Added

- Detailed logging output for MICE imputation when `verbose` is set to 3 or higher, including iteration progress and timing information.
- Successful completion message for MICE imputation when `verbose` is set to 2 or higher.


## [0.2.0] - 2026-08-19

### Added

- Multithreading support utilizing the `rayon` crate to parallelize row-level matrix building and data preparation.
- Full support for categorical feature imputation utilizing smartcore's `RandomForestClassifier`.
- `cat_cols` parameter in the Python API to accept a sequence of categorical column indices.
- `seed` parameter to guarantee reproducible deterministic outputs while preserving statistical variance via dynamic offsetting.
- `verbose` parameter to allow users to control the level of logging output during execution.
- `n_jobs` parameter to allow users strict control over the maximum number of CPU cores utilized during execution.

### Changed

- Locked Rust dependency versions in `Cargo.toml` (replacing wildcards) to guarantee long-term build stability.


## [0.1.2] - 2026-08-18

### Added

- Added `CHANGELOG.md` file to document changes and updates in the project.
- Added `dragon_imputation.pyi` Python type stubs for improved IDE support, docstrings, and type checking.

### Changed

- `mice_impute()` function signature updated to replace `pydf` argument with `df` for clarity and consistency.


## [0.1.1] - 2026-08-18

### Fixed

- Removed unsupported Linux architectures (`s390x`, `ppc64le`) from GitHub Actions CI matrix to resolve build failures.


## [0.1.0] - 2026-08-18

### Added

- Initial release of `dragon-imputation`.
- Core MICE imputation algorithm backed by Rust.
- Implemented Random Forest Regressor for imputing missing numerical values.
- Polars DataFrame integration via `pyo3-polars`.
- Automated CI/CD pipeline for building and publishing to PyPI via Trusted Publisher Management.
