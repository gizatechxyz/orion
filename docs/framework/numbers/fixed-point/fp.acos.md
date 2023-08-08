# fp.acos

```rust
fn acos(self: FixedType) -> FixedType;
```

Returns the  arccosine (inverse of cosine) of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the asin  of the input value.

## Examples

```rust
fn acos_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(1);

// We can call `asin` function as follows.
fp.acos()
}
>>> {mag: 0, sign: true} // = 0...
```
