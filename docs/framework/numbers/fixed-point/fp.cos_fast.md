# fp.cos_fast

```rust
fn cos_fast(self: FixedType) -> FixedType;
```

Returns the cosine of the fixed point number fast with LUT.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the cosine of the input value.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn cos_fast_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `cos_fast` function as follows.
    fp.cos_fast()
}
>>> {mag: 27273, sign: true} // = -0.41614684
``` 
