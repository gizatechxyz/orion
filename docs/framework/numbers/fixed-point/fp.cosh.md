# fp.cosh

```rust
fn cosh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic cosine of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn cosh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `cosh` function as follows.
    fp.cosh()
}
>>> {mag: 246559, sign: false} // = 3.76219569
``` 
