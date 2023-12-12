# ComplexTrait::log10

```rust
fn log10(self: T) -> T;
```

Returns the base-10 logarithm of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the base 10 logarithm of the input number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn log10_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.log10()
}
>>> {real: {mag: 10274314139629458970, sign: false}, im: {mag: 7873411322133748801, sign: false}} // 0.5569716761 + 0.4268218908 i
 ```
