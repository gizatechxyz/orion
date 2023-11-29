# ComplexTrait::new

```rust
fn new(real: F, img: F) -> T;
```

## Args

* `real`(`F`) - The real part of the complex number.
* `img`(`F`) - The imaginary part of the complex number.

## Returns

A new complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};


fn new_complex64_example() -> complex64 {
    ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false))
}
>>> {real: {mag: 184467440737095516160, sign: false}, im: {mag: 18446744073709551616, sign: false}} // 10 + i
```
