# int.div_rem

```rust
fn div_rem(self: T, other: T) -> (T, T);
```

Computes signed\_integer division and modulus simultaneously

## Args

* `self`(`T`) - The dividend
* `other`(`T`) - The divisor

## Panics

Panics if the divisor is zero.

## Returns

A tuple of signed integer `<T>`, containing the quotient and the remainder of the division.

## Examples

```rust
fn div_rem_example() -> (i32, i32) {
// We instantiate signed integers here.
let a = IntegerTrait::<i32>::new(13, false);
let b = IntegerTrait::<i32>::new(5, false);

// We can call `div_rem` function as follows.
a.div_rem(b)
}
>>> ({mag: 2, sign: false}, {mag: 3, sign: false}) // = (2, 3)
```
