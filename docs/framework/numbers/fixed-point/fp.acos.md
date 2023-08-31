# fp.acos

```rust
fn acos(self: T) -> T;
```

Returns the  arccosine (inverse of cosine) of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the acos  of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn acos_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `acos` function as follows.
    fp.acos()
}
>>> {mag: 205887, sign: false} // = 3.14159265
``` 
