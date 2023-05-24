
```rust
fn softsign(tensor: @Tensor<T>) -> Tensor<FixedType>;
```

Applies the Softsign function to an n-dimensional input Tensor such that the elements of the n-dimensional output Tensor lie in the range \[-1,1].

$$
\text{softsign}(x_i) = \frac{x_i}{1 + |x_i|}
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32;

fn softsign_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `softsign` function as follows.
return NNTrait::softsign(@tensor);
}
>>> [[0,33554432],[44739242,50331648]]
// The fixed point representation of
// [[0, 0.5],[0.67, 0.75]]
```
