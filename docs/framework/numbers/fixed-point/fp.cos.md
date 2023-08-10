# fp.cos

```rust
fn cos(self: FixedType) -> FixedType;
```

Returns the cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the cosine of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn cos_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `cos` function as follows.
    fp.cos()
}
>>> {mag: 27273, sign: true} // = -0.41614684
``` 
