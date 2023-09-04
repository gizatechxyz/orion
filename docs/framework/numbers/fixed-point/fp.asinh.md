# fp.asinh

```rust
fn asinh(self: T) -> T;
```

Returns the value of the inverse hyperbolic sine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The inverse hyperbolic sine of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn asinh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `asinh` function as follows.
    fp.asinh()
}
>>> {mag: 94610, sign: false} // = 1.44363548
``` 
