# NN::relu

Applies the rectified linear unit function element-wise

$$
ReLU(x)=(x)^+=max(0,x)
$$

```rust
fn relu(tensor: @Tensor<T>) -> Tensor<T>;
```

#### Args

| Name     | Type         | Description       |
| -------- | ------------ | ----------------- |
| `tensor` | `@Tensor<T>` | The input tensor. |

> _`<T>` generic type depends on NN dtype._

#### Returns

A `Tensor<T>` with the same shape as the input tensor.

> _`<T>` generic type depends on NN dtype._

#### Examples

```rust
use onnx_cairo::operators::nn::nn_i32::NN;

fn relu_example() -> Tensor<u32> {
    // We instantiate a 2D Tensor here.
    // [[1,2],[-1,-2]]
    let tensor = u32_tensor_2x2_helper();
		
    // We can call `relu` function as follows.
    return NN::relu(@tensor);
}
>>> [[1,2],[0,0]]
```
