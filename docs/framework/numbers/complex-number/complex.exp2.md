# ComplexTrait::exp2

```rust
fn exp2(self: T) -> T;
```

Returns the value of 2 raised to the power of the complex number.

## Args

* `self`(`T`) - The input complex number

## Returns

The binary exponent of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn exp2_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    ComplexTrait::exp2(z)
}
>>> {real: {mag: 197471674372309809080, sign: true}, im: {mag: 219354605088992285353, sign: true}} // -10.70502356986 -11.89127707 i
``` 
