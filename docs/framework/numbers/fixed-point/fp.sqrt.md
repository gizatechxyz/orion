# fp.sqrt

```rust
fn sqrt(self: T) -> T;
```

Returns the square root of the fixed point number.

## Args

`self`(`T`) - The input fixed point

## Panics

* Panics if the input is negative.

## Returns

A fixed point number representing the square root of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn sqrt_fp_example() -> FP16x16 {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::new_unscaled(9, false);

    // We can call `round` function as follows.
    a.sqrt()
}
>>> {mag: 196608, sign: false} // = 3
```
