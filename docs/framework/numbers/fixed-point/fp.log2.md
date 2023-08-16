# fp.log2

```rust
fn log2(self: FixedType) -> FixedType;
```

Returns the base-2 logarithm of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Panics

* Panics if the input is negative.

## Returns

A fixed point representing the binary logarithm of the input number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn log2_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(3, false);

    // We can call `log2` function as follows.
    fp.log2()
}
>>> {mag: 103872, sign: false} // = 1.58496250072
```
