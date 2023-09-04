# fp.log10

```rust
fn log10(self: T) -> T;
```

Returns the base-10 logarithm of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point representing the base 10 logarithm of the input number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn log10_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(3, false);

    // We can call `log10` function as follows.
    fp.log10()
}
>>> {mag: 31269, sign: false} // = 0.47712125472
```
