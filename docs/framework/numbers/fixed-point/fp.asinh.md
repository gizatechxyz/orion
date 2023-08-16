# fp.asinh

```rust
fn asinh(self: FixedType) -> FixedType;
```

Returns the value of the inverse hyperbolic sine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The inverse hyperbolic sine of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn asinh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `asinh` function as follows.
    fp.asinh()
}
>>> {mag: 94610, sign: false} // = 1.44363548
``` 
