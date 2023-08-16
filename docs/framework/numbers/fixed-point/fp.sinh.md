# fp.sinh

```rust
fn sinh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic sine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic sine of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn sinh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sinh` function as follows.
    fp.sinh()
}
>>> {mag: 237690, sign: false} // = 3.62686041
``` 
