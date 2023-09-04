# fp.acos_fast

```rust
fn acos_fast(self: T) -> T;
```

Returns the  arccosine (inverse of cosine) of the fixed point number faster with LUT.

## Args

* `self`(`T`) - The input fixed point

## Returns

A fixed point number representing the acos  of the input value.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn acos_fast_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(1, true);

    // We can call `acos_fast` function as follows.
    fp.acos_fast()
}
>>> {mag: 205887, sign: false} // = 3.14159265
``` 
