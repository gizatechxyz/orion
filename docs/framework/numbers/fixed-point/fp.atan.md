# fp.atan

```rust
fn atan(self: T) -> T;
```

Returns the arctangent (inverse of tangent) of the input fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn atan_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `atan` function as follows.
    fp.atan()
}
>>> {mag: 72558, sign: false} // = 1.10714872
``` 
 