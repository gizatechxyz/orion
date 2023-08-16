# NNTrait::softsign

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
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
use orion::numbers::fixed_point::core::{FixedImpl, FixedType};

fn softsign_example() -> Tensor<FixedType> {
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

    return NNTrait::softsign(@tensor);
}
>>> [[0,4194304],[5592405,6291456]]
    // The fixed point representation of
    // [[0, 0.5],[0.67, 0.75]]
```
