# fp.atan

```rust
fn atan(self: FixedType) -> FixedType;
```

Returns the arctangent (inverse of tangent) of the input fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn atan_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `atan` function as follows.
    fp.atan()
}
>>> {mag: 72558, sign: false} // = 1.10714872
``` 
 