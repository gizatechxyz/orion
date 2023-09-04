# fp.tanh

```rust
fn tanh(self: T) -> T;
```

Returns the value of the hyperbolic tangent of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The hyperbolic tangent of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn tanh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `tanh` function as follows.
    fp.tanh()
}
>>> {mag: 63179, sign: false} // = 0.96402758
``` 
