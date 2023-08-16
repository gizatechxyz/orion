# fp.atanh

```rust
fn atanh(self: FixedType) -> FixedType;
```

Returns the value of the inverse hyperbolic tangent of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The inverse hyperbolic tangent of the input fixed point number.

## Examples

```rust
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;

fn atanh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(32768); // 0.5

    // We can call `atanh` function as follows.
    fp.atanh()
}
>>> {mag: 35999, sign: false} // = 0.54930614
``` 
