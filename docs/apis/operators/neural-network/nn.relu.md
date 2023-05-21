# NNTrait::relu

```rust
fn relu(tensor: @Tensor<T>) -> Tensor<T>;
```

Applies the rectified linear unit function element-wise

$$
ReLU(x)=(x)^+=max(0,x)
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A `Tensor<T>` with the same shape as the input tensor.

## Examples

```rust
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::implementations::impl_nn_i32;


fn relu_example() -> Tensor<u32> {
// We instantiate a 2D Tensor here.
// [[1,2],[-1,-2]]
let tensor = u32_tensor_2x2_helper();

// We can call `relu` function as follows.
return NNTrait::relu(@tensor);
}
>>> [[1,2],[0,0]]
```
