# fp.exp

```rust
fn exp(self: FixedType) -> FixedType;
```

Returns the value of e raised to the power of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The natural exponent of the input fixed point number.

## Examples

```rust
fn exp_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(2);
    
    // We can call `exp` function as follows.
    fp.exp()
}
>>> {mag: 495871144, sign: false} // = 7.389056317241236
``` 
