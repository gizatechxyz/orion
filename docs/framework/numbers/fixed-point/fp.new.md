# FixedTrait::new

```rust
fn new(mag: MAG, sign: bool) -> T;
```

Constructs a new fixed point instance.

## Args

* `mag`(`MAG`) - The magnitude of the fixed point.
* `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn new_fp_example() -> FP16x16 {
    // We can call `new` function as follows. 
    FixedTrait::new(65536, false)
}
>>> {mag: 65536, sign: false} // = 1 in FP16x16
```
