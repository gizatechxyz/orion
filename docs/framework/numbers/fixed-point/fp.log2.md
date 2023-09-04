# fp.log2

```rust
fn log2(self: T) -> T;
```

Returns the base-2 logarithm of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Panics

* Panics if the input is negative.

## Returns

A fixed point representing the binary logarithm of the input number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn log2_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(3, false);

    // We can call `log2` function as follows.
    fp.log2()
}
>>> {mag: 103872, sign: false} // = 1.58496250072
```
