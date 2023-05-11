# fp.ln

Returns the natural logarithm of the fixed point number.

```rust
fn ln(self: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description           |
| ------ | ----------- | --------------------- |
| `self` | `FixedType` | The input fixed point |

#### Returns

A FixedType value representing the natural logarithm of the input number.

#### Examples

```rust
fn ln_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = Fixed::from_unscaled_felt(1);
    
    // We can call `ln` function as follows.
    fp.ln()
}
>>> {mag: 0, sign: false}
```
