# ComplexTrait::reciprocal


```rust
fn reciprocal(self: T) -> T;
```

Returns a the reciprocal of the complex number (i.e. 1/z).

## Args

* `self`(`T`) - The input complex number.

## Returns 

The reciprocal of the complex number \(a + bi\) is given by:
\[
\frac{1}{a + bi} = \frac{a}{a^2 + b^2} - \frac{b}{a^2 + b^2}i
\]

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn reciprocal_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.reciprocal()
}
>>> {real: {mag: 41453357469010228, sign: false}, im: {mag: 435260253424607397, sign: true}} // 0.002247191011 - 0.0235955056 i
 ```
