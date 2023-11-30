# ComplexTrait::arg

```rust
fn arg(self: T) -> F;
```

Returns the argument of the complex number

## Args

* `self`(`T`) - The input complex number

## Returns

A fixed point number '<F>', representing the argument of the complex number in radian. 
'arg(z) = atan2(b, a)'.

## Examples

```rust    
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn arg_complex64_example() -> FP64x64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.arg()
}
>>> {mag: 27224496882576083824, sign: false} // arg = 1.4758446204521403 (rad)
```
