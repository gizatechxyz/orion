# fp.round

Rounds the fixed point number to the nearest whole number.

```rust
fn round(self: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description           |
| ------ | ----------- | --------------------- |
| `self` | `FixedType` | The input fixed point |

#### Returns

A fixed point number representing the rounded value.

#### Examples

```rust
fn round_fp_example() -> FixedType {
    // We instantiate fixed points here.
    let a = Fixed::from_felt(194615506); // 2.9
    
    // We can call `round` function as follows.
    a.round(b)
}
>>> {mag: 201326592, sign: false} // = 3
```
