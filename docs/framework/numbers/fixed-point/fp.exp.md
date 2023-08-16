# fp.exp

```rust
fn exp(self: FixedType) -> FixedType;
```

Returns the value of e raised to the power of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The natural exponent of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn exp_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `exp` function as follows.
    fp.exp()
}
>>> {mag: 484249, sign: false} // = 7.389056317241236
``` 
