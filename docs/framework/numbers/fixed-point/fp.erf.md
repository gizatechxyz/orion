# fp.erf

```rust
fn erf(self: T) -> T;
```

Returns the error function of the input fixed point number computed element-wise.

## Args

* `self`(`T`) - The input fixed point

## Returns

The error function of the input fixed point number computed element-wise.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn erf_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::new(65536, false);

    // We can call `erf` function as follows.
    fp.erf()
}
>>> {mag: 55227, sign: false} // = -1
``` 
