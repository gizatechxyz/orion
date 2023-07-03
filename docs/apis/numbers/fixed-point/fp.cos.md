# fp.cos

```rust
fn cos(self: FixedType) -> FixedType;
```

Returns the cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the cosine of the input value.

## Examples

```rust
fn cos_fp_example() -> FixedType {
// We instantiate fixed point here.
let fp = FixedTrait::from_unscaled_felt(2);

// We can call `cos` function as follows.
fp.cos()
}
>>> {mag: 27273, sign: true} // = -0.4161..
```
