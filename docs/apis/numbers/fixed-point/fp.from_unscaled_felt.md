# FixedTrait::from\_unscaled\_felt

```rust
fn from_unscaled_felt(val: felt252) -> FixedType<T>;
```

Creates a new fixed point instance from an unscaled felt252 value.

## Args

`val`(`felt252`) - `felt252` value to convert in FixedType<T>

## Returns - A new fixed point instance.

## Examples

```rust
fn from_unscaled_felt_example() -> FixedType<T> {
// We can call `from_unscaled_felt` function as follows .
FixedTrait::from_unscaled_felt(1);
}
>>> {mag: 67108864, sign: false}
```
