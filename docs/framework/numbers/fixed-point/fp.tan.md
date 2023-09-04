# fp.tan

```rust
fn tan(self: T) -> T;
```

Returns the tangent of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the tan of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn tan_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `tan` function as follows.
    fp.tan()
}
>>> {mag: 143199, sign: true} // = -2.18503986
``` 
