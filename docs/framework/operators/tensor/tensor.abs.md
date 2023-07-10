#tensor.abs

```rust
fn abs(self: @Tensor<T>) -> Tensor<T>;
```

Computes the absolute value of all elements in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the absolute value of all elements in the input tensor.

## Example

```rust
fn abs_example() -> Tensor<i32> {
// We instantiate a 3D Tensor here.
// tensor = [[0,-1,2],[-3,4,5],[-6,7,-8]]
let tensor = i32_tensor_3x3x3_helper();
let result = tensor.abs();
return result;
}
>>> [0,1,2,3,4,5,6,7,8]
```
