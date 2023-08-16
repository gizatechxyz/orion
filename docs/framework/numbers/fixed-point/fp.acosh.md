# fp.acosh

```rust
fn acosh(self: FixedType) -> FixedType;
```

Returns the value of the inverse hyperbolic cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The inverse hyperbolic cosine of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn acosh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `acosh` function as follows.
    fp.acosh()
}
>>> {mag: 86308, sign: false} // = 1.3169579
``` 
