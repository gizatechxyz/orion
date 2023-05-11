# fp.log10

Returns the base-10 logarithm of the fixed point number.

```rust
fn log10(self: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description           |
| ------ | ----------- | --------------------- |
| `self` | `FixedType` | The input fixed point |

#### Returns

A FixedType value representing the base 10 logarithm of the input number.

#### Examples

```rust
fn log10_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = Fixed::from_unscaled_felt(100);
    
    // We can call `log10` function as follows.
    fp.log10()
}
>>> {mag: 134217717, sign: false} // = 1.9999999873985543
```
