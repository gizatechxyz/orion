# fp.log10

```rust
fn log10(self: FixedType<T>) -> FixedType<T>;
```

Returns the base-10 logarithm of the fixed point number.

## Args

* `self`(`FixedType<T>`) - The input fixed point

## Returns

A fixed point representing the base 10 logarithm of the input number.

## Examples

```rust
fn log10_fp_example() -> FixedType<T> {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(100);

// We can call `log10` function as follows.
fp.log10()
}
>>> {mag: 134217717, sign: false} // = 1.9999999873985543
```
