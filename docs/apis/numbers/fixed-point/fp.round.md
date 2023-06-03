# fp.round

```rust
fn round(self: FixedType<T>) -> FixedType<T>;
```

Rounds the fixed point number to the nearest whole number.

## Args

* `self`(`FixedType<T>`) - The input fixed point

## Returns

A fixed point number representing the rounded value.

## Examples


```rust
fn round_fp_example() -> FixedType<T> {
// We instantiate FixedTrait points here.
let a = FixedTrait::from_felt(194615506); // 2.9

// We can call `round` function as follows.
a.round(b)
}
>>> {mag: 201326592, sign: false} // = 3
```
