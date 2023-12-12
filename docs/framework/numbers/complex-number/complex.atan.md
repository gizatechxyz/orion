# ComplexTrait::atan

```rust
fn atan(self: T) -> T;
```

Returns the arctangent (inverse of tangent) of the input complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn atan_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.atan()
}
>>> {real: {mag: 26008453796191787243, sign: false}, im: {mag: 4225645162986888119, sign: false}} // 1.40992104959 + 0.2290726829i
 ```
 