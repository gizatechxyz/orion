#tensor.tanh

```rust
fn tanh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the hyperbolic tangent of each element in the input tensor.

## Args

- `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the hyperbolic tangent of the values in the input tensor.

## Example

```rust
fn tanh_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0,1,2, -1, -2]]
// for each value val we calculate hyperbolic tangent with the formula:
// (exp(val)+exp(-val))/(exp(val)-exp(-val))
let result = tensor.tanh();
return result;
}
>>> [0, 0.76159, 0.96403, -0.76159, -0.96403]
```
