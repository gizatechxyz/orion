# fp.ln


```rust
fn ln(self: FixedType) -> FixedType;
```

Returns the natural logarithm of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns 

A fixed point representing the natural logarithm of the input number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn ln_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, false);

    // We can call `ln` function as follows.
    fp.ln()
}
>>> {mag: 0, sign: false}
```
