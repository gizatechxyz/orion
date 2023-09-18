# FixedTrait::new\_unscaled

```rust
    fn new_unscaled(mag: MAG, sign: bool) -> T;
```

Creates a new fixed point instance with the specified unscaled magnitude and sign.

## Args

`mag`(`MAG`) - The unscaled magnitude of the fixed point.
`sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn new_unscaled_example() -> FP16x16 {
    // We can call `new_unscaled` function as follows. 
    FixedTrait::new_unscaled(1, false)
}
>>> {mag: 65536, sign: false}
```
