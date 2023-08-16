# fp.sin

```rust
fn sin(self: FixedType) -> FixedType;
```

Returns the sine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the sin of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn sin_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sin` function as follows.
    fp.sin()
}
>>> {mag: 59592, sign: false} // = 0.90929743
``` 
