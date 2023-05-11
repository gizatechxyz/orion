# fp.sqrt

Returns the square root of the fixed point number.

```rust
fn sqrt(self: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description           |
| ------ | ----------- | --------------------- |
| `self` | `FixedType` | The input fixed point |

#### Panics

| TypeError                        |
| -------------------------------- |
| Panics if the input is negative. |

#### Returns

A fixed point number representing the square root of the input value.

#### Examples

```rust
fn sqrt_fp_example() -> FixedType {
    // We instantiate fixed points here.
    let a = Fixed::from_unscaled_felt(25);
    
    // We can call `round` function as follows.
    a.sqrt()
}
>>> {mag: 1677721600, sign: false} // = 5
```
