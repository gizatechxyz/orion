# int.min

```rust
fn min(self: T, other: T) -> T;
```

Returns the minimum between two signed\_integer.

## Args

`self`(`T`) - The first signed integer to compare.
`other`(`T`) - The second signed integer to compare.

## Returns

A signed integer `<T>`, The minimum between `self` and `other`.

## Examples


```rust
fn min_example() -> i32 {
// We instantiate signed integer here.
let a = IntegerTrait::<i32>::new(42_u32, true);
let b = IntegerTrait::<i32>::new(13_u32, false);

// We can call `max` function as follows.
a.min(b)
}
>>> {mag: 42, sign: true} // as -42 < 13
```
