# ComplexTrait::acos

```rust
fn acos(self: T) -> T;
```

Returns the  arccosine (inverse of cosine) of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns

A complex number representing the acos  of the input value.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn acos_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.acos()
}
>>> {real: {mag: 18449430688981877061, sign: false}, im: {mag: 36587032881711954470, sign: true}} //  1.000143542473797 - 1.98338702991653i
 ```
