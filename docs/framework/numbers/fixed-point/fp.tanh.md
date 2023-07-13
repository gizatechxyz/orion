# fp.tanh

```rust
fn tanh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic tangent of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic tangent of the input fixed point number.

## Examples

```rust
fn tanh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(2);
    
    // We can call `tanh` function as follows.
    fp.tanh()
}
>>> {mag: 8086850, sign: false} // = 0.964027...
``` 
