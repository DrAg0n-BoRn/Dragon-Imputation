# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

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
