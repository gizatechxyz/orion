# ComplexTrait::atanh

```rust
fn atanh(self: T) -> T;
```

Returns the value of the inverse hyperbolic tangent of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

The inverse hyperbolic tangent of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn atanh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.atanh()
}
>>> {real: {mag: 2710687792925618924, sign: false}, im: {mag: 24699666646262346226, sign: false}} //  0.146946666 + 1.33897252i
 ```

