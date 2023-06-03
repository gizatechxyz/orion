# fp.ln


```rust
fn ln(self: FixedType<T>) -> FixedType<T>;
```

Returns the natural logarithm of the fixed point number.

## Args

* `self`(`FixedType<T>`) - The input fixed point

## Returns

A fixed point representing the natural logarithm of the input number.

## Examples

```rust
fn ln_fp_example() -> FixedType<T> {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(1);

// We can call `ln` function as follows.
fp.ln()
}
>>> {mag: 0, sign: false}
```
