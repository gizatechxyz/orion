# ComplexTrait::cos

```rust
fn cos(self: T) -> T;
```

Returns the cosine of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the cosine of the input value.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn cos_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.cos()
}
>>> {real: {mag: 77284883172661882094, sign: true}, im: {mag: 168035443352962049425, sign: true}} // -4.18962569 + -9.10922789375i
 ```

