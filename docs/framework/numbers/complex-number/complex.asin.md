# ComplexTrait::asin

```rust
fn asin(self: T) -> T;
```

Returns the  arcsine (inverse of sine) of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the asin of the input value.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn asin_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.asin()
}
>>> {real: {mag: 10526647143326614308, sign: false}, im: {mag: 36587032881711954470, sign: false}} // 0.57065278432 + 1.9833870299i
 ```
