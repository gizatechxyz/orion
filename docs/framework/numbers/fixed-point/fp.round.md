# fp.round

```rust
fn round(self: T) -> T;
```

Rounds the fixed point number to the nearest whole number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the rounded value.

## Examples


```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn round_fp_example() -> FP16x16 {
    // We instantiate FixedTrait points here.
    let a = FixedTrait::from_felt(190054); // 2.9

    // We can call `round` function as follows.
    a.round()
}
>>> {mag: 196608, sign: false} // = 3
```
