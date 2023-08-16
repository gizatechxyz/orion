# NNTrait::logsoftmax

```rust 
   fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>
```

Applies the natural log to Softmax function to an n-dimensional input Tensor consisting of values in the range \[0,1].

$$
\text{log softmax}(x_i) = \log(frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}})
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the natural lof softmax outputs.

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

fn logsoftmax_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(0, false),
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(3, false),
        ].span(),
        extra: Option::Some(extra)
    );

    return NNTrait::logsoftmax(@tensor, 1);
}
    This will first generate the softmax output tensor
>>> [[2255697,6132911],[2255697,6132911]]
    // The fixed point representation of
    // [[0.2689, 0.7311],[0.2689, 0.7311]]
    
    Applying the natural log to this tensor yields
>>> 
    // The fixed point representation of:
    // [[-1.3134, -0.3132],[-1.3134, -0.3132]]
```
