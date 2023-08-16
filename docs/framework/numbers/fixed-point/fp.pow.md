# fp.pow

```rust
fn pow(self: FixedType, b: FixedType) -> FixedType;
```

Returns the result of raising the fixed point number to the power of another fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point.
* `b`(`FixedType`) - The exponent fixed point number.

## Returns

A fixed point number representing the result of x^y.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn pow_fp_example() -> FixedType {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::new_unscaled(3, false);
    let b = FixedTrait::new_unscaled(4, false);

    // We can call `pow` function as follows.
    a.pow(b)
}
>>> {mag: 5308416, sign: false} // = 81
```
