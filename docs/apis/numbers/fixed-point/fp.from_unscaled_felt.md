# Fixed::from\_unscaled\_felt

```rust
fn from_unscaled_felt(val: felt252) -> FixedType;
```

Creates a new FixedType instance from an unscaled felt252 value.

## Args

`val`(`felt252`) - `felt252` value to convert in FixedType

## Returns - A new `FixedType` instance.

## Examples

```rust
fn from_unscaled_felt_example() -> FixedType {
// We can call `from_unscaled_felt` function as follows .
Fixed::from_unscaled_felt(1);
}
>>> {mag: 67108864, sign: false}
```
