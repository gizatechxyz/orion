# fp.pow

```rust
fn pow(self: T, b: T) -> T;
```

Returns the result of raising the fixed point number to the power of another fixed point number.

## Args

* `self`(`T`) - The input fixed point.
* `b`(`T`) - The exponent fixed point number.

## Returns

A fixed point number representing the result of x^y.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn pow_fp_example() -> FP16x16 {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::new_unscaled(3, false);
    let b = FixedTrait::new_unscaled(4, false);

    // We can call `pow` function as follows.
    a.pow(b)
}
>>> {mag: 5308416, sign: false} // = 81
```
