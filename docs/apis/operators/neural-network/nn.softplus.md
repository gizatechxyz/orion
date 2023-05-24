
```rust
fn softplus(tensor: @Tensor<T>) -> Tensor<FixedType>;
```

Applies the Softplus function to an n-dimensional input Tensor such that the elements of the n-dimensional output Tensor lie in the range \[-1,1].

$$
\text{softplus{x_i} = \ln(1 + e^{x_i})
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32;

fn softplus_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `softplus` function as follows.
return NNTrait::softplus(@tensor);
}
>>> [[46516187,88131451],[142735719,204587229]]
// The fixed point representation of
// [[0.6931452, 1.31326096],[2.12692796, 3.04858728]]
```
