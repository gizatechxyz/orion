# fp.ceil

```rust
fn ceil(self: FixedType) -> FixedType;
```

Returns the smallest integer greater than or equal to the fixed point number.

## Args

*`self`(`FixedType`) - The input fixed point

## Returns

The smallest integer greater than or equal to the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn ceil_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(190054); // 2.9

    // We can call `ceil` function as follows.
    fp.ceil()
}
>>> {mag: 196608, sign: false} // = 3
```
