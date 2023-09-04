# fp.asin

```rust
fn asin(self: T) -> T;
```

Returns the  arcsine (inverse of sine) of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the asin of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn asin_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, false);

    // We can call `asin` function as follows.
    fp.asin()
}
>>> {mag: 102943, sign: true} // = 1.57079633
``` 
