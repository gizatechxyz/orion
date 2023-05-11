# Fixed::new

Constructs a new FixedType instance.

```rust
fn new(mag: u128, sign: bool) -> FixedType;
```

#### Args

| Name   | Type   | Description                                                             |
| ------ | ------ | ----------------------------------------------------------------------- |
| `mag`  | `u128` | The magnitude of the fixed point.                                       |
| `sign` | `bool` | The sign of the fixed point, where `true` represents a negative number. |

#### Returns

A new `FixedType` instance.

#### Examples

```rust
fn new_fp_example() -> FixedType {
    // We can call `new` function as follows. 
    Fixed::new(67108864, false)
}
>>> {mag: 67108864, sign: false} // = 1
```
