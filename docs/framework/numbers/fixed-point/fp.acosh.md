# fp.acosh

```rust
fn acosh(self: T) -> T;
```

Returns the value of the inverse hyperbolic cosine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The inverse hyperbolic cosine of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn acosh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `acosh` function as follows.
    fp.acosh()
}
>>> {mag: 86308, sign: false} // = 1.3169579
``` 
