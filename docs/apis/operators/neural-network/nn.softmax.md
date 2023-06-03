# NNTrait::softmax

```rust
fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType<F>>;
```

Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1] and sum to 1.

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the softmax.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32;

fn softmax_example() -> Tensor<FixedType<F>> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `softmax` function as follows.
return NNTrait::softmax(@tensor, 1);
}
>>> [[2255697,6132911],[2255697,6132911]]
// The fixed point representation of
// [[0.2689, 0.7311],[0.2689, 0.7311]]
```
