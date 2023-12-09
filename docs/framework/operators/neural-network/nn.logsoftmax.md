# NNTrait::logsoftmax

```rust 
   fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<T>
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

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
use orion::operators::nn::{NNTrait, FP8x23NN};
use orion::numbers::{FP8x23, FixedTrait};

fn logsoftmax_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new(0, false),
            FixedTrait::new(1, false),
            FixedTrait::new(2, false),
            FixedTrait::new(3, false),
        ]
            .span(),
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
