# fp.sin

```rust
fn sin(self: T) -> T;
```

Returns the sine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the sin of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn sin_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sin` function as follows.
    fp.sin()
}
>>> {mag: 59592, sign: false} // = 0.90929743
``` 
