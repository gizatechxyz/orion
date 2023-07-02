# fp.asinh

```rust
fn asinh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The inverse hyperbolic sine of the input fixed point number.

## Examples

```rust
fn asinh_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `asinh` function as follows.
fp.asinh()
}
>>> {mag: 12110093, sign: false} // = 1.443635...
```
