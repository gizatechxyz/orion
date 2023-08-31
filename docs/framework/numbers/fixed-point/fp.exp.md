# fp.exp

```rust
fn exp(self: T) -> T;
```

Returns the value of e raised to the power of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The natural exponent of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn exp_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `exp` function as follows.
    fp.exp()
}
>>> {mag: 484249, sign: false} // = 7.389056317241236
``` 
