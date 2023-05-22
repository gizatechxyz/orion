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
* `threshold`(`T`) - A scalar that defines the threshold below which the Relu function returns 0.

## Returns

A `Tensor<T>` with the same shape as the input tensor.

## Examples

```rust
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::implementations::impl_nn_i32;

fn relu_example() -> Tensor<u32> {
// We instantiate a 2D Tensor here and set threshold to 0.
// [[1,2],[-1,-2]]
let tensor = i32_tensor_2x2_helper();
let threshold = IntegerTrait::new(0, false);

// We can call `relu` function as follows.
return NNTrait::relu(@tensor, threshold);
}
>>> [[1,2],[0,0]]
```
