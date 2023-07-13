# fp.asin

```rust
fn asin(self: FixedType) -> FixedType;
```

Returns the  arcsine (inverse of sine) of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns

A fixed point number representing the asin  of the input value.

## Examples

```rust
fn asin_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(1);
    
    // We can call `asin` function as follows.
    fp.asin()
}
>>> {mag: 102943, sign: true} // = 1.5707...
``` 
