# fp.abs

```rust
fn abs(self: T) -> T;
```

Returns the absolute value of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The absolute value of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};


fn abs_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `abs` function as follows.
    fp.abs()
}
>>> {mag: 65536, sign: false} // = 1
```
