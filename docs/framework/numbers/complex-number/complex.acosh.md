# ComplexTrait::acosh

```rust
fn acosh(self: T) -> T;
```

Returns the value of the inverse hyperbolic cosine of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

The inverse hyperbolic cosine of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn acosh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.acosh()
}
>>> {real: {mag: 36587032878947915965, sign: false}, im: {mag: 18449360714192945790, sign: false}} // 1.9833870 + 1.0001435424i
 ```
