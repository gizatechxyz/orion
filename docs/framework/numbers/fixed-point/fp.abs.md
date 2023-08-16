# fp.abs

```rust
fn abs(self: FixedType) -> FixedType;
```

Returns the absolute value of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The absolute value of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn abs_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `abs` function as follows.
    fp.abs()
}
>>> {mag: 65536, sign: false} // = 1
```
