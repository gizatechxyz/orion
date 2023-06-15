#tensor.cosh

```rust
fn cosh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the hyperbolic cosine of each element in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the hyperbolic cosine of the values in the input tensor.

## Example

```rust
fn cosh_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0,1,2]]
let tensor = fp8x23_tensor_1x3_helper();
let result = tensor.cosh();
return result;
}
>>> [1,1.5403,3.7622]
```
