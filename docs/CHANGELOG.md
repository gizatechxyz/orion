# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2023-05-07
### Changed

- Update code structure.

## [Unreleased] - 2023-05-05
### Changed

- Upgrade code to cairo alpha-v1.0.0-rc0:
  - update syntax to fit with cairo alpha-v1.0.0-rc0 
  - refactor imports

## [Unreleased] - 2023-05-05

### Added

- Added exponential element-wise operation
- Added softmax 

## [Unreleased] - 2023-05-04

### Fixed

- Broadcast operations (add/sub/mul/div)


## [Unreleased] - 2023-05-02

### Added

- Implemented Fixed Point Q5.26
- Implemented Tensor with Fixed Type

## [Unreleased] - 2023-05-01

### Changed

- Remove duplicate files in docs.

## [Unreleased] - 2023-05-01

### Changed

- Removed Matrix and Vector types.
- Refactored functions to be Tensor-centric.
- Reorganize code structure.
- Temporarily remove softmax.

## [Unreleased] - 2023-04-29

### Added

- Tensor `matmul` feature:
  - Implemented matrix multiplication for tensors.
  - Behavior depends on the dimensionality of the tensors:
    - 1D-1D: Returns the dot product.
    - 2D-2D: Returns the matrix-matrix product.
    - 1D-2D: Returns the matrix-vector product after temporarily prepending a 1 to the 1D tensor's dimension.
    - 2D-1D: Returns the matrix-vector product.
  - Matmul for more than 2D tensors is not supported.

## [Unreleased] - 2023-04-25

### Changed

- Tensor optimization and refactoring:
  - Replaced `@Array` with `Span` for improved performance and compatibility.
  - Changed iteration method from `span.at()` to `span.pop_front()` for better efficiency.
  - No breaking changes or API changes introduced.

## [Unreleased] - 2023-04-24

### Changed

- Tensor code improvements and documentation updates:
  - Added missing checks in the Tensor implementation for better robustness.
  - Refactored the code to use Rust-style comments for better readability and consistency.

## [Unreleased] - 2023-04-22

### Added

- Tensor reshape and transpose features:
  - Implemented `tensor.reshape` for changing the shape of a tensor.
  - Implemented `tensor.transpose` according to the given axes permutation.
  - Provided descriptions for the new functions.
  - Created unit tests for the new features.

## [Unreleased] - 2023-04-14

### Changed

- Refactored loop implementation:
  - Replaced recursive functions with built-in loop expressions for better performance and readability.
  - Changed the type of Tensor shape to `Span<usize>` for improved compatibility and efficiency.
  - Added alpha7 artifact.
  - No breaking changes or API changes introduced.

## [Unreleased] - 2023-04-12

### Changed

- Refactored signed integer implementation:
  - Improved compatibility with existing regimes (i8, i32, i64, i128).
  - Implemented i8, i32, i64, i128 with magnitudes bounded by 2**(n-1).
  - No breaking changes or API changes introduced.

## [Unreleased] - 2023-03-31

### Added

- Tensor implementation in Cairo 1.0:
  - Implemented TensorTrait.
  - Added broadcast element-wise operations.
  - Added reduce sum operation.
  - Added argmax operation.
  - Created unit tests for the new features.
- Replaced Vector and Matrix with Tensor object as an nd-array in the future (no breaking change in this release).

### Changed

- Updated the behavior of the library to use the new Tensor object for enhanced functionality.

### Removed

- Deprecated Vector and Matrix for future removal (no breaking change in this release).