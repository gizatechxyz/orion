# ComplexTrait::conjugate

```rust
fn conjugate(self: T) -> T;
```
  
Returns the conjugate of a complex number. The complex number is represented in Cartesian form `z = a + bi`.
The conjugate of `z = a + bi` is `z̅ = a - bi`

## Args

* `self`(`T`) - The complex number from which we want the conjugate.

## Returns

A complex number `<T>`, representing the imaginary part of `self` .

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn conjugate_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    z.conjugate()
}
>>> {real: {mag: 184467440737095516160, sign: false}, im: {mag: 18446744073709551616, sign: true}} // 10 - i
```
