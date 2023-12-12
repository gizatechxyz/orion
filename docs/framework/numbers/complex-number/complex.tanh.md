# ComplexTrait::tanh

```rust
fn tanh(self: T) -> T;
```

Returns the value of the hyperbolic tangent of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

The hyperbolic tangent of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn tanh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.tanh()
}
>>> {real: {mag: 17808227710002974080, sign: false}, im: {mag: 182334107030204896, sign: true}} // 0.96538587902 + 0.009884375i
 ```
