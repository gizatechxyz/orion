# ComplexTrait::log2

```rust
fn log2(self: T) -> T;
```

Returns the base-2 logarithm of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Panics

* Panics if the input is negative.

## Returns

A complex number representing the binary logarithm of the input number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn log2_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.log2()
}
>>> {real: {mag: 34130530934667840346, sign: false}, im: {mag: 26154904847122126193, sign: false}} // 1.85021986 + 1.41787163 i
 ```
