# fp.pow

Returns the result of raising the fixed point number to the power of another fixed point number.

```rust
fn pow(self: FixedType, b: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description                      |
| ------ | ----------- | -------------------------------- |
| `self` | `FixedType` | The input fixed point            |
| `b`    | `FixedType` | The exponent fixed point number. |

#### Returns

A fixed point number representing the result of x^y.

#### Examples

```rust
fn pow_fp_example() -> FixedType {
    // We instantiate fixed points here.
    let a = Fixed::from_unscaled_felt(3); 
    let b = Fixed::from_unscaled_felt(4);
    
    // We can call `pow` function as follows.
    a.pow(b)
}
>>> {mag: 5435817984, sign: false} // = 81
```
