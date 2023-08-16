# FixedTrait::new

```rust
fn new(mag: u32, sign: bool) -> FixedType;
```

Constructs a new fixed point instance.

## Args

* `mag`(`u32`) - The magnitude of the fixed point.
* `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn new_fp_example() -> FixedType {
    // We can call `new` function as follows. 
    FixedTrait::new(65536, false)
}
>>> {mag: 65536, sign: false} // = 1 in FP16x16
```
