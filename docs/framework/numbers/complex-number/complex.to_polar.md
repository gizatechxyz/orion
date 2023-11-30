
//fn log2(self: T) -> T;

//fn log10(self: T) -> T;

# ComplexTrait::to_polar

```rust
fn to_polar(self: T) -> (F, F);
```

Returns the polar coordinates (magnitude and argument) of the complex number.

## Args

* `self`(`T`) - The input complex number.

## Returns 

A tuple of two fixed point numbers representing the polar coordinates of the input number.

## Examples

```rust
use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};

fn to_polar_complex64_example() -> (FP64x64, FP64x64) {
    let z: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false),
        FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    z.to_polar()
}
>>> ({mag: 778268985067028086784, sign: false},  {mag: 27224496882576083824, sign: false}) // mag : 42.190046219457976 + arg : 1.4758446204521403 
 ```
