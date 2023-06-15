#tensor.sinh

```rust
fn sinh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the hyperbolic sine of each element in the input tensor.

## Args

- `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the hyperbolic sine of the values in the input tensor.

## Example

```rust
fn sinh_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0,1,2]]
// for each value val we calculate hyperbolic sine with the formula:
// (exp(val)-exp(-val))/2
let tensor = fp8x23_tensor_1x3_helper();
let result = tensor.sinh();
return result;
}
>>> [0, 1.1752, 3.6269]
```
