# ComplexTrait::mag

```rust
fn mag(self: T) -> F;
```

Returns the magnitude of the complex number

## Args

* `self`(`T`) - The input complex number

## Returns

A fixed point number '<F>', representing the magnitude of the complex number. 
'mag(z) = sqrt(a^2 + b^2)'.

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn mag_complex64_example() -> FP64x64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.mag()
}
>>> {mag: 0x2a30a6de7900000000, sign: false} // mag = 42.190046219457976
```
