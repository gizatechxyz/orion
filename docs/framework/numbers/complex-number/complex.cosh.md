# ComplexTrait::cosh

```rust
fn cosh(self: T) -> T;
```

Returns the value of the hyperbolic cosine of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

The hyperbolic cosine of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn cosh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.cosh()
}
>>> {real: {mag: 68705646899632870392, sign: true}, im: {mag: 9441447324287988702, sign: false}} // -3.72454550491 + 0.511822569987i
 ```
