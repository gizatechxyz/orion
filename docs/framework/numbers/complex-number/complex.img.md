# ComplexTrait::img

```rust
fn img(self: T) -> F;
```

Returns the imaginary part of a complex number. The complex number is represented in Cartesian form `z = a + bi` where `b` is the imaginary part.

## Args

* `self`(`T`) - The complex number from which we want the imaginary part.

## Returns

A fixed point number `<F>`, representing the imaginary part of `self` .

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn img_complex64_example() -> FP64x64 {
    let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    z.img()
}
>>> {mag: 18446744073709551616, sign: false} // 1
```
