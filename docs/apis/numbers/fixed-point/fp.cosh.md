# fp.cosh

```rust
fn cosh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic cosine of the fixed point number.

## Args

- `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic cosine of the input fixed point number.

## Examples

```rust
fn cosh_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `cosh` function as follows.
fp.cosh()
}
>>> {mag: 31559577, sign: false} // = 3.62686...
```
