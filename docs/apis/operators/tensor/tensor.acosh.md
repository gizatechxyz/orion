#tensor.acosh

```rust
fn acosh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the inverse hyperbolic cosine of each element in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the inverse hyperbolic cosine of the values in the input tensor.

## Example

```rust
fn acosh_example() -> Tensor<FixedType> {
We instantiate a 1D Tensor here.
tensor = [[0, 1, 2]]
// for each value val we calculate inverse hyperbolic sine with the formula:
// ln( x + sqrt( 1 - x^2))
// for all x >= 1
let result = tensor.acosh();
return result;
}
>>> [PANIC!, 0, 1.31696]
```
