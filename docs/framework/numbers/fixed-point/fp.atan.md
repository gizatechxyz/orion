# fp.atan

```rust
fn atan(self: FixedType) -> FixedType;
```

Returns the arctangent (inverse of tangent) of the input fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the arctangent (inverse of tangent) of the input value.

## Examples

```rust
fn atan_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `atan` function as follows.
fp.atan()
}
>>> {mag: 72558, sign: false} // = 1.1071...
```
