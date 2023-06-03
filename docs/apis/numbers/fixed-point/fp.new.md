# FixedTrait::new

```rust
fn new(mag: u128, sign: bool) -> FixedType<T>;
```

Constructs a new fixed point instance.

## Args

* `mag`(`u128`) - The magnitude of the fixed point.
* `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.

## Returns

A new fixed point instance.

## Examples

```rust
fn new_fp_example() -> FixedType<T> {
// We can call `new` function as follows.
FixedTrait::new(67108864, false)
}
>>> {mag: 67108864, sign: false} // = 1
```
