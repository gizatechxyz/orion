# ComplexTrait::from_polar


```rust
fn from_polar(mag: F, arg: F) -> T;
```

Returns a complex number (in the Cartesian form) from the polar coordinates of the complex number.

## Args

* `mag`(`F`) - The input fixed point number representing the magnitude.
* `arg`(`F`) - The input fixed point number representing the argument.

## Returns 

The complex number representing the Cartesian form calculated from the input polar coordinates.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn from_polar_complex64_example() -> complex64 {
    let mag: FP64x64 = FixedTrait::new(778268985067028086784, false); // 42.190046219457976
    let arg: FP64x64 = FixedTrait::new(27224496882576083824, false); //1.4758446204521403
    ComplexTrait::from_polar(mag,arg)
}
>>> {real: {mag: 73787936714814843012, sign: false}, im: {mag: 774759489569697723777, sign: false}} // 4 + 42 i
 ```
