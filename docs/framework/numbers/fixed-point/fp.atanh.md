# fp.atanh

```rust
fn atanh(self: T) -> T;
```

Returns the value of the inverse hyperbolic tangent of the fixed point number.

## Args

* `self`(`T`) - The input fixed point

## Returns

The inverse hyperbolic tangent of the input fixed point number.

## Examples

```rust
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};

fn atanh_fp_example() -> FP16x16 {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_felt(32768); // 0.5

    // We can call `atanh` function as follows.
    fp.atanh()
}
>>> {mag: 35999, sign: false} // = 0.54930614
``` 
