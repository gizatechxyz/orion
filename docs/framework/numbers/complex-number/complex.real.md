# ComplexTrait::real

```rust
fn real(self: T) -> F;
```

Returns the real part of a complex number. The complex number is represented in Cartesian form `z = a + bi` where `a` is the real part.

## Args

* `self`(`T`) - The complex number from which we want the real part.

## Returns

A fixed point number `<F>`, representing the real part of `self` .

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn real_complex64_example() -> FP64x64 {
    let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    z.real()
}
>>> {mag: 184467440737095516160, sign: false} // 10
```
