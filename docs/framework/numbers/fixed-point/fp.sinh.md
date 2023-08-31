# fp.sinh

```rust
fn sinh(self: T) -> T;
```

Returns the value of the hyperbolic sine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The hyperbolic sine of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn sinh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `sinh` function as follows.
    fp.sinh()
}
>>> {mag: 237690, sign: false} // = 3.62686041
``` 
