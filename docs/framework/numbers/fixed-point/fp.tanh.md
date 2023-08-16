# fp.tanh

```rust
fn tanh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic tangent of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic tangent of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn tanh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `tanh` function as follows.
    fp.tanh()
}
>>> {mag: 63179, sign: false} // = 0.96402758
``` 
