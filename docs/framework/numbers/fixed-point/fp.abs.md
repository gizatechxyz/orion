# fp.abs

```rust
fn abs(self: FixedType) -> FixedType;
```

Returns the absolute value of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The absolute value of the input fixed point number.

## Examples

```rust
fn abs_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(-1);
    
    // We can call `abs` function as follows.
    fp.abs()
}
>>> {mag: 67108864, sign: false} // = 1
```
