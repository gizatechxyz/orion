# fp.sin

```rust
fn sin(self: FixedType) -> FixedType;
```

Returns the sine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the sin  of the input value.

## Examples

```rust
fn sin_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(2);
    
    // We can call `sin` function as follows.
    fp.sin()
}
>>> {mag: 59592, sign: false} // = 0.909..
``` 
