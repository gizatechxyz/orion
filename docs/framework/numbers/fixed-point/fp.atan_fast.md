# fp.atan_fast

```rust
fn atan_fast(self: T) -> T;
```

Returns the arctangent (inverse of tangent) of the input fixed point number faster with LUT.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn atan_fast_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `atan_fast` function as follows.
    fp.atan_fast()
}
>>> {mag: 72558, sign: false} // = 1.10714872
``` 
