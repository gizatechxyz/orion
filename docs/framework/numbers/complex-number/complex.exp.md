# ComplexTrait::exp

```rust
fn exp(self: T) -> T;
```

Returns the value of e raised to the power of the complex number.

## Args

* `self`(`T`) - The input complex number

## Returns

The natural exponent of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn exp_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    ComplexTrait::exp(z)
}
>>> {real: {mag: 402848450095324460000, sign: true}, im: {mag: 923082101320478400000, sign: true}} // -21.838458238788455-50.04038098170736 i
``` 
