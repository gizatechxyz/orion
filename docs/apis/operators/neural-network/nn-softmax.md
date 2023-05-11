# NN::softmax

Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1] and sum to 1.

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

```rust
fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
```

Values returned between 0 and 1 are represented as [Fixed Points](../../numbers/fixed-point/) in this implementation.

#### Args

| Name     | Type         | Description                                  |
| -------- | ------------ | -------------------------------------------- |
| `tensor` | `@Tensor<T>` | The input tensor.                            |
| `axis`   | `usize`      | The axis along which to compute the softmax. |

> _`<T>` generic type depends on NN dtype._

#### Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

#### Examples

```rust
use onnx_cairo::operators::nn::nn_i32::NN;

fn softmax_example() -> Tensor<FixedType> {
    // We instantiate a 2D Tensor here.
    // [[0,1],[2,3]]
    let tensor = u32_tensor_2x2_helper();
		
    // We can call `softmax` function as follows.
    return NN::softmax(@tensor, 1);
}
>>> [[18048353,49060510],[18048352,49060511]]
    // The fixed point representation of
    // [[0.2689, 0.7311],[0.2689, 0.7311]]
```
