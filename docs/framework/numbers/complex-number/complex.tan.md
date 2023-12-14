# ComplexTrait::tan

```rust
fn tan(self: T) -> T;
```

Returns the tangent of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the tan of the input value.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn tan_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.tan()
}
>>> {real: {mag: 69433898428143694, sign: true}, im: {mag: 18506486100303669886, sign: false}} // -0.00376402 + 1.00323862i
 ```
