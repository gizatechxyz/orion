# fp.exp2

```rust
fn exp2(self: T) -> T;
```

Returns the value of 2 raised to the power of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The binary exponent of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn exp2_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `exp2` function as follows.
    fp.exp2()
}
>>> {mag: 262143, sign: false} // = 3.99999957248
``` 
