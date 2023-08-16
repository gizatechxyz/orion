# fp.acos

```rust
fn acos(self: FixedType) -> FixedType;
```

Returns the  arccosine (inverse of cosine) of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the acos  of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn acos_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `acos` function as follows.
    fp.acos()
}
>>> {mag: 205887, sign: false} // = 3.14159265
``` 
