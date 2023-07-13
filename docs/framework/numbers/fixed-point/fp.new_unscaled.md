# FixedTrait::new\_unscaled

```rust
fn new_unscaled(mag: u128, sign: bool) -> FixedType;
```

Creates a new fixed point instance with the specified unscaled magnitude and sign.

## Args

`mag`(`u128`) - The unscaled magnitude of the fixed point.
`sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
fn new_unscaled_example() -> FixedType {
    // We can call `new_unscaled` function as follows. 
    FixedTrait::new_unscaled(1);
}
>>> {mag: 67108864, sign: false}
```
