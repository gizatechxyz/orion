# fp.tan_fast

```rust
fn tan_fast(self: FixedType) -> FixedType;
```

Returns the tangent of the fixed point number faster with LUT.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the tan of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn tan_fast_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `tan_fast` function as follows.
    fp.tan_fast()
}
>>> {mag: 143199, sign: true} // = -2.18503986
``` 
