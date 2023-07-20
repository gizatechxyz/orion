# fp.acosh

```rust
fn acosh(self: FixedType) -> FixedType;
```

Returns the value of the inverse hyperbolic cosine of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

The inverse hyperbolic cosine of the input fixed point number.

## Examples

```rust
fn acosh_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(2);
    
    // We can call `acosh` function as follows.
    fp.acosh()
}
>>> {mag: 11047444, sign: false} // = 1.31696...
``` 
