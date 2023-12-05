# ComplexTrait::one

```rust
fn one(self: T) -> T;
```
  
Returns the multiplicative identity element one

## Returns

A complex number `<T>`, representing the multiplicative identity element of the complex field : `1 + 0i`. 

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn one_complex64_example() -> complex64 {
    ComplexTrait::one()
}
>>> {real: {mag: 18446744073709551616, sign: false}, im: {mag: 0, sign: false}} // 1 + 0i
```
