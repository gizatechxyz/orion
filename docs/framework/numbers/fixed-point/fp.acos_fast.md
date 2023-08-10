# fp.acos_fast

```rust
fn acos_fast(self: FixedType) -> FixedType;
```

Returns the  arccosine (inverse of cosine) of the fixed point number faster with LUT.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the acos  of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn acos_fast_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `acos_fast` function as follows.
    fp.acos_fast()
}
>>> {mag: 205887, sign: false} // = 3.14159265
``` 
