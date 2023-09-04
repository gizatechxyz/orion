# fp.floor

```rust
fn floor(self: T) -> T;
```

Returns the largest integer less than or equal to the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

Returns the largest integer less than or equal to the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn floor_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(190054); // 2.9

    // We can call `floor` function as follows.
    fp.floor()
}
>>> {mag: 131072, sign: false} // = 2
```
