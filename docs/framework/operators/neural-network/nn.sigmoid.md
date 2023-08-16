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
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
use orion::numbers::fixed_point::core::{FixedImpl, FixedType};

fn sigmoid_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(0, false),
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(3, false),
        ]
            .span(),
        extra: Option::Some(extra)
    );

    return NNTrait::sigmoid(@tensor);
}
>>> [[4194304,6132564],[7388661,7990771]]
    // The fixed point representation of
    // [[0.5, 0.7310586],[0.88079703, 0.95257413]]
```
