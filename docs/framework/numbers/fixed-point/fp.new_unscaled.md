# FixedTrait::new\_unscaled

```rust
    fn new_unscaled(mag: u32, sign: bool) -> FixedType;
```

Creates a new fixed point instance with the specified unscaled magnitude and sign.

## Args

`mag`(`u32`) - The unscaled magnitude of the fixed point.
`sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn new_unscaled_example() -> FixedType {
    // We can call `new_unscaled` function as follows. 
    FixedTrait::new_unscaled(1, false)
}
>>> {mag: 65536, sign: false}
```
