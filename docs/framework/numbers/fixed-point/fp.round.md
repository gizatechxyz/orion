# fp.round

```rust
fn round(self: FixedType) -> FixedType;
```

Rounds the fixed point number to the nearest whole number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the rounded value.

## Examples


```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn round_fp_example() -> FixedType {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::from_felt(190054); // 2.9

    // We can call `round` function as follows.
    a.round()
}
>>> {mag: 196608, sign: false} // = 3
```
