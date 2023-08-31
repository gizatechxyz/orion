# fp.cosh

```rust
fn cosh(self: T) -> T;
```

Returns the value of the hyperbolic cosine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The hyperbolic cosine of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn cosh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `cosh` function as follows.
    fp.cosh()
}
>>> {mag: 246559, sign: false} // = 3.76219569
``` 
