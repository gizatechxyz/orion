# fp.exp2

```rust
fn exp2(self: FixedType) -> FixedType;
```

Returns the value of 2 raised to the power of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The binary exponent of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn exp2_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `exp2` function as follows.
    fp.exp2()
}
>>> {mag: 262143, sign: false} // = 3.99999957248
``` 
