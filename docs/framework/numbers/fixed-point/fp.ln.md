# fp.ln


```rust
fn ln(self: T) -> T;
```

Returns the natural logarithm of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns 

A fixed point representing the natural logarithm of the input number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn ln_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, false);

    // We can call `ln` function as follows.
    fp.ln()
}
>>> {mag: 0, sign: false}
```
