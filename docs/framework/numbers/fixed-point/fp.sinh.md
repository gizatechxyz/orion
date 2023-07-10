# fp.sinh

```rust
fn sinh(self: FixedType) -> FixedType;
```

Returns the value of the hyperbolic sine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The hyperbolic sine of the input fixed point number.

## Examples

```rust
fn sinh_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `sinh` function as follows.
fp.sinh()
}
>>> {mag: 30424311, sign: false} // = 3.6268604
```
