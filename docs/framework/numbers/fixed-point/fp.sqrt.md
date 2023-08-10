# fp.sqrt

```rust
fn sqrt(self: FixedType) -> FixedType;
```

Returns the square root of the fixed point number.

## Args

`self`(`FixedType`) - The input fixed point

## Panics

* Panics if the input is negative.

## Returns

A fixed point number representing the square root of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn sqrt_fp_example() -> FixedType {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::new_unscaled(9, false);

    // We can call `round` function as follows.
    a.sqrt()
}
>>> {mag: 196608, sign: false} // = 3
```
