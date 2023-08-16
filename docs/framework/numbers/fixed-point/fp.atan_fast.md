# fp.atan_fast

```rust
fn atan_fast(self: FixedType) -> FixedType;
```

Returns the arctangent (inverse of tangent) of the input fixed point number faster with LUT.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn atan_fast_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `atan_fast` function as follows.
    fp.atan_fast()
}
>>> {mag: 72558, sign: false} // = 1.10714872
``` 
