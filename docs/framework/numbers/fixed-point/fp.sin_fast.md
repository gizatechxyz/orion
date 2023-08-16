# fp.sin_fast

```rust
fn sin_fast(self: FixedType) -> FixedType;
```

Returns the sine of the fixed point number faster with LUT.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the sin of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn sin_fast_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sin_fast` function as follows.
    fp.sin_fast()
}
>>> {mag: 59592, sign: false} // = 0.90929743
``` 
