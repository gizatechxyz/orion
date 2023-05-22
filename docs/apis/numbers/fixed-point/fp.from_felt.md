# Fixed::from\_felt


```rust
fn from_felt(val: felt252) -> FixedType;
```

Creates a new FixedType instance from a felt252 value.

## Args

* `val`(`felt252`) - `felt252` value to convert in FixedType

## Returns

A new `FixedType` instance.

## Examples

```rust
fn from_felt_example() -> FixedType {
// We can call `from_felt` function as follows .
Fixed::from_felt(194615706);
}
>>> {mag: 194615706, sign: false} // = 2.9
```
