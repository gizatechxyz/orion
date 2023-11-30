# ComplexTrait::zero

```rust
fn zero(self: T) -> T;
```
  
Returns the additive identity element zero

## Returns

A complex number `<T>`, representing the additive identity element of the complex field `0`.

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn zero_complex64_example() -> complex64 {
    ComplexTrait::zero()
}
>>> {real: {mag: 0, sign: false}, im: {mag: 0, sign: false}} // 0 + 0i
```
