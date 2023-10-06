# fp.sign

```rust
fn sign(self: T) -> T;
```

Returns the element-wise indication of the sign of the input fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The element-wise indication of the sign of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn sign_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new_unscaled(2, true);

    // We can call `sign` function as follows.
    fp.sign()
}
>>> {mag: 65536, sign: true} // = -1
``` 
