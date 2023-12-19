# ComplexTrait::asinh

```rust
fn asinh(self: T) -> T;
```

Returns the value of the inverse hyperbolic sine of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

The inverse hyperbolic sine of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn asinh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.asinh()
}
>>> {real: {mag: 36314960239770126586, sign: false}, im: {mag: 17794714057579789616, sign: false}} //1.9686379 + 0.964658504i
 ```
