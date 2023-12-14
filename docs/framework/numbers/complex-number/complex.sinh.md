# ComplexTrait::sinh

```rust
fn sinh(self: T) -> T;
```

Returns the value of the hyperbolic sine of the complex number.
## Args

* `self`(`T`) - The input complex number.

## Returns

The hyperbolic sine of the input complex number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn sinh_complex64_example() -> complex64 {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false),
        FixedTrait::new(55340232221128654848, false)
    ); // 2 + 3i
    z.sinh()
}
>>> {real: {mag: 66234138518106676624, sign: true}, im: {mag: 9793752294470951790, sign: false}} // -3.59056458998 + 0.530921086i
 ```
