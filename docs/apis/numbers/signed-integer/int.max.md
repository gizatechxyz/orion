# int.max

```rust
fn max(self: T, other: T) -> T;
```

Returns the maximum between two signed\_integer.

## Args

*`self`(`T`) - The first signed integer to compare.
* `other`(`T`) - The second signed integer to compare.

## Returns

A signed integer `<T>`, The maximum between `self` and `other`.

## Examples

```rust
fn max_example() -> i32 {
// We instantiate signed integer here.
let a = IntegerTrait::<i32>::new(42_u32, true);
let b = IntegerTrait::<i32>::new(13_u32, false);

// We can call `max` function as follows.
a.max(b)
}
>>> {mag: 13, sign: false} // as 13 > -42
```
