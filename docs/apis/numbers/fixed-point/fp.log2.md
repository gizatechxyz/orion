# fp.log2

```rust
fn log2(self: FixedType<T>) -> FixedType<T>;
```

Returns the base-2 logarithm of the fixed point number.

## Args

* `self`(`FixedType<T>`) - The input fixed point

## Panics

* Panics if the input is negative.

## Returns

A fixed point representing the binary logarithm of the input number.

## Examples

```rust
fn log2_fp_example() -> FixedType<T> {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(32);

// We can call `log2` function as follows.
fp.log2()
}
>>> {mag: 335544129, sign: false} // = 4.99999995767848
```
