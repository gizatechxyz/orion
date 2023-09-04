# fp.ceil

```rust
fn ceil(self: T) -> T;
```

Returns the smallest integer greater than or equal to the fixed point number.

## Args

*`self`(`T`) - The input fixed point

## Returns

The smallest integer greater than or equal to the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn ceil_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(190054); // 2.9

    // We can call `ceil` function as follows.
    fp.ceil()
}
>>> {mag: 196608, sign: false} // = 3
```
