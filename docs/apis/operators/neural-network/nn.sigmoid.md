# NNTrait::sigmoid

```rust
fn sigmoid(tensor: @Tensor<T>) -> Tensor<FixedType>;
```

Applies the Sigmoid function to an n-dimensional input tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1].

$$
\text{sigmoid}(x_i) = \frac{1}{1 + e^{-x_i}}
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

fn sigmoid_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `sigmoid` function as follows.
return NNTrait::sigmoid(@tensor);
}
>>> [[4194304,6132564],[7388661,7990771]]
// The fixed point representation of
// [[0.5, 0.7310586],[0.88079703, 0.95257413]]
```
