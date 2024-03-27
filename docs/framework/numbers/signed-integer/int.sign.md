# int.sign

```rust
fn sign(self: T, other: T) -> T;
```

Returns an element-wise indication of the given signed_integer.

## Args

`self`(`T`) - The input value to which the signed value is applied.

## Returns

An element-wise indication of the sign of a number.

## Examples


```rust
fn sign_example() -> i32 {
    // We instantiate signed integer here.
    let a = IntegerTrait::<i32>::new(42, true);
    
    // We can call `sign` function as follows.
    a.sign()
}
>>> {mag: 1, sign: true}
```
