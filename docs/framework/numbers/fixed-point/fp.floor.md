# fp.floor

```rust
fn floor(self: FixedType) -> FixedType;
```

Returns the largest integer less than or equal to the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

Returns the largest integer less than or equal to the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn floor_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(190054); // 2.9

    // We can call `floor` function as follows.
    fp.floor()
}
>>> {mag: 131072, sign: false} // = 2
```
