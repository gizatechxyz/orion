# Fixed::new\_unscaled

Creates a new FixedType instance with the specified unscaled magnitude and sign.

```rust
fn new_unscaled(mag: u128, sign: bool) -> FixedType;
```

#### Args

| Name   | Type   | Description                                                             |
| ------ | ------ | ----------------------------------------------------------------------- |
| `mag`  | `u128` | The unscaled magnitude of the fixed point.                              |
| `sign` | `bool` | The sign of the fixed point, where `true` represents a negative number. |

#### Returns

A new `FixedType` instance.

#### Examples

```rust
fn new_unscaled_example() -> FixedType {
    // We can call `new_unscaled` function as follows. 
    Fixed::new_unscaled(1);
}
>>> {mag: 67108864, sign: false}
```
