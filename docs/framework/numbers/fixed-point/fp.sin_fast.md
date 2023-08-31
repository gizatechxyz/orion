# fp.sin_fast

```rust
fn sin_fast(self: T) -> T;
```

Returns the sine of the fixed point number faster with LUT.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the sin of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn sin_fast_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sin_fast` function as follows.
    fp.sin_fast()
}
>>> {mag: 59592, sign: false} // = 0.90929743
``` 
