# ComplexTrait::pow

```rust
fn pow(self: T, b: T) -> T;
```

Returns the result of raising the complex number to the power of another complex number.

## Args

* `self`(`T`) - The input complex number.
* `b`(`T`) - The exponent complex number.

## Returns

A complex number representing the result of z^w.

## Examples

```rust
use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::complex_number::complex64::{TWO, complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn pow_2_complex64_example() -> complex64 {
    let two = ComplexTrait::new(FP64x64Impl::new(TWO, false),FP64x64Impl::new(0, false));
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.pow(two)
}
>>> {real: {mag: 32244908640844296224768, sign: true}, im: {mag: 6198106008766409342976, sign: false}} // -1748 + 336 i

fn pow_w_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i

    let w: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(18446744073709551616, false)
    ); // 2 + i
    z.pow(w)
}
>>> {real: {mag: 6881545343236111419203, sign: false}, im: {mag: 2996539405459717736042, sign: false}} // -373.0485407816205 + 162.4438823807959 i
```
