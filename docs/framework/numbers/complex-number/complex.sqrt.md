# ComplexTrait::sqrt

```rust
fn arg(self: T) -> F;
```

Returns the value of the squre root of the complex number.

## Args

* `self`(`T`) - The input complex number

## Returns

A complex number '<T>', representing the square root of the complex number. 
'arg(z) = atan2(b, a)'.

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn sqrt_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.sqrt()
}
>>> {real: {mag: 88650037379463118848, sign: false}, im: {mag: 80608310115317055488, sign: false}} // 4.80572815603723 + 4.369785247552674 i
```
