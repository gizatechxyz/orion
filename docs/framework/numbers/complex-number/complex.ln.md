# ComplexTrait::ln

```rust
fn ln(self: T) -> T;
```

Returns the natural logarithm of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns 

A complex number representing the natural logarithm of the input number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn ln_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.ln()
}
>>> {real: {mag: 69031116512113681970, sign: false}, im: {mag: 27224496882576083824, sign: false}} // 3.7421843216430655 + 1.4758446204521403 i
 ```
