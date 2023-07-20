# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2023-07-20

### Added
- Added arccosine (acos) implementation and tests

## [Unreleased] - 2023-07-11

### Changed
- Refactored Perfomance trait to take two generic parameters: 
  - `T`, the type of the unquantized tensor. Currently support `FixedType` and `i32`.
  - `Q`, the type of the quantized tensor. Currently support `i8`.


## [Unreleased] - 2023-07-10

### Added
- Implement Tensor int8
- Implement NN int8
- Added DequantizeLinear operator

### Fixed
- Fixed docgen

## [Unreleased] - 2023-07-04

### Added
- Added arcsin implementation and tests
  
## [Unreleased] - 2023-07-08

### Added

- Added arctangent operator
## [Unreleased] - 2023-07-06

### Added

- Add QuantizeLinear

## [Unreleased] - 2023-07-01

### Added
- Added cosine implementation and tests

## [Unreleased] - 2023-06-29

### Added

- Added acosh operator

## [Unreleased] - 2023-06-29

### Added

- Added flatten operator

## [Unreleased] - 2023-06-28

### Added

- Added asinh operator

## [Unreleased] - 2023-06-28

### Added

- Added cosh operator

## [Unreleased] - 2023-06-27

### Added

- Added tanh operator

## [Unreleased] - 2023-06-27

### Added

- Added sinh operator

## [Unreleased] - 2023-06-24

### Added

- Added cumsum operator

## [Unreleased] - 2023-06-23

### Changed

- Replace array.at with IndexView
  
## [Unreleased] - 2023-06-30

### Added
- Added sine implementation and tests

## [Unreleased] - 2023-06-21

### Changed

- Refactor nn tests to cover n-dimensions

## [Unreleased] - 2023-06-21

### Changed

- Refactor tensor tests to cover n-dimensions

## [Unreleased] - 2023-06-20

### Changed

- Upgrade Cairo version to v2.0.0-rc

## [Unreleased] - 2023-06-17

### Added

- Into trait to signed integers

## [Unreleased] - 2023-06-16

### Changed

- Updated argmax function parameters
- Restructured tests for argmax and argmin

## [Unreleased] - 2023-06-11

### Changed

- Added ln functionality to tensor trait. Added Logsoftmax implementation for nn trait.
- Added tests for both.

## [Unreleased] - 2023-06-08

### Added

- Added argmin tensor operator

## [Unreleased] - 2023-06-07

### Changed

- Remove the check range from the creation of a new fixed point,
  so that we can use fp for wide calculations.
  A future PR will restore the check range with a better design.

## [Unreleased] - 2023-06-07

### Added

- Added ceil operator

### Fixed

- Fixed abs tensor operator doc

## [Unreleased] - 2023-06-06

### Added

- Add fixed-point Q16.16 implementation

### Changed

- Refactor operators and performance functions to support multiple fixed-point implementations.

## [Unreleased] - 2023-06-05

### Added

- extra parameter to Tensor.

## [Unreleased] - 2023-06-03

### Changed

- Refactor fixed point to support multiple implementations

### Fixed

- Fix greater operator.
- Update compatibility.md

## [Unreleased] - 2023-06-02

### Fixed

- Fix reduce_sum on 1D Tensor.

## [Unreleased] - 2023-06-01

### Added

- Added greater tensor operator and tests
- Added less tensor operator and tests

## [Unreleased] - 2023-05-30

### Added

- Added abs tensor operator and tests
- Added sigmoid operator and tests

## [Unreleased] - 2023-05-27

### Added

- Added equality (element-wise) operator and tests

### Fixed

- Fixed imports to allow tests in linear_test.cairo work

## [Unreleased] - 2023-05-24

### Added

- Added softsign implementation and tests

## [Unreleased] - 2023-05-25

### Added

- Add Linear Quantization from FixedType Tensor

## [Unreleased] - 2023-05-25

### Changed

- Fix docgen.

## [Unreleased] - 2023-05-25

### Changed

- Replace Fixed Point Q5.26 with Q8.23

## [Unreleased] - 2023-05-24

### Added

- Added softplus implementation and tests

## [Unreleased] - 2023-05-22

### Changed

- Remove unused variables.
- Renaming. Replace ONNX-Cairo with Orion.

## [Unreleased] - 2023-05-20

### Changed

- Updated relu u32 function to have a threshold parameter
- Updated relu test folder structure

## [Unreleased] - 2023-05-19

### Added

- Added leaky relu
- Updated NN test folder structure

## [Unreleased] - 2023-05-19

### Added

- Added range check to FixedType

## [Unreleased] - 2023-05-19

### Added

- Added range check to FixedType

## [Unreleased] - 2023-05-21

### Add

- `docgen` script to generate documentation from docstrings.

### Changed

- NN and Performance modules to NNTrait and PerformanceTrait

## [Unreleased] - 2023-05-11

### Added

- Added linear layer (dtype: Tensor<i32>, Tensor<u32>)

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
  - Implemented i8, i32, i64, i128 with magnitudes bounded by 2\*\*(n-1).
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
