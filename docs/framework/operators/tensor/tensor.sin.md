#tensor.sin

```rust
fn sin(self: @Tensor<T>) -> Tensor<T>;
```

Computes the sine of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the sine value of all elements in the input tensor.

## Example

```rust
fn sin_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0, 1, 2,]]
let tensor = fp8x23_tensor_1x3_helper();
let result = tensor.sin();
return result;
}
>>> [0,7058770,7627740]
// The fixed point representation of
// [0,0.8414...,0.9092...]
```
