# fp.log


```rust
fn log(self: FixedType) -> FixedType;
```

Returns the natural logarithm of the fixed point number.

## Args

* `self`(`FixedType`) - The input fixed point

## Returns 

A fixed point representing the natural logarithm of the input number.

## Examples

```rust
fn log_fp_example() -> FixedType {
    // We instantiate fixed point here.
    let fp = FixedTrait::from_unscaled_felt(1);
    
    // We can call `log` function as follows.
    fp.log()
}
>>> {mag: 0, sign: false}
```
