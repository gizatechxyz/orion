# fp.cos

```rust
fn cos(self: T) -> T;
```

Returns the cosine of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the cosine of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn cos_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, false);

    // We can call `cos` function as follows.
    fp.cos()
}
>>> {mag: 27273, sign: true} // = -0.41614684
``` 
