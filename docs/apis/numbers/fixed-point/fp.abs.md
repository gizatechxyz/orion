# fp.abs

Returns the absolute value of the fixed point number.

```rust
fn abs(self: FixedType) -> FixedType;
```

#### Args

| Name   | Type        | Description           |
| ------ | ----------- | --------------------- |
| `self` | `FixedType` | The input fixed point |

#### Returns

The absolute value of the input fixed point number.

#### Examples

```rust
fn abs_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = Fixed::from_unscaled_felt(-1);
    
    // We can call `abs` function as follows.
    fp.abs()
}
>>> {mag: 67108864, sign: false} // = 1
```
