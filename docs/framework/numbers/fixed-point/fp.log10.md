# fp.log10

```rust
fn log10(self: FixedType) -> FixedType;
```

Returns the base-10 logarithm of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point representing the base 10 logarithm of the input number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn log10_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(3, false);

    // We can call `log10` function as follows.
    fp.log10()
}
>>> {mag: 31269, sign: false} // = 0.47712125472
```
