# Fixed::from\_unscaled\_felt

Creates a new FixedType instance from an unscaled felt252 value.

```rust
fn from_unscaled_felt(val: felt252) -> FixedType;
```

#### Args

| Name  | Type      | Description                             |
| ----- | --------- | --------------------------------------- |
| `val` | `felt252` | `felt252` value to convert in FixedType |

#### Returns

A new `FixedType` instance.

#### Examples

```rust
fn from_unscaled_felt_example() -> FixedType {
    // We can call `from_unscaled_felt` function as follows . 
    Fixed::from_unscaled_felt(1);
}
>>> {mag: 67108864, sign: false}
```
