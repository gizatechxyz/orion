# fp.exp2

```rust
fn exp2(self: FixedType) -> FixedType;
```

Returns the value of 2 raised to the power of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The binary exponent of the input fixed point number.

## Examples

```rust
fn exp2_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `exp2` function as follows.
fp.exp2()
}
>>> {mag: 268435456, sign: false} // = 3.99999957248
```
