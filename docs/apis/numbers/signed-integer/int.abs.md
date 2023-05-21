# int.abs

```rust
fn abs(self: T) -> T;
```

Computes the absolute value of a signed\_integer.

## Args

`self`(`T`) - The signed integer to which the absolute value is applied

## Returns

A signed integer `<T>`, representing the absolute value of `self` .

## Examples

```rust
fn abs_example() -> i32 {
// We instantiate signed integers here.
let int = IntegerTrait::<i32>::new(42_u32, true);

// We can call `abs` function as follows.
a.abs()
}
>>> {mag: 42, sign: false} // = 42
```
