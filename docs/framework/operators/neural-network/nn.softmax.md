# NNTrait::softmax

```rust 
   fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<F>;
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
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_i32_fp8x23};
use orion::operators::nn::{NNTrait, NN_i32_fp8x23};
use orion::numbers::{i32, FP8x23, IntegerTrait};

fn softmax_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(0, false),
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(3, false),
        ]
            .span(),
    );

    return NNTrait::softmax(@tensor, 1);
}
>>> [[2255697,6132911],[2255697,6132911]]
    // The fixed point representation of
    // [[0.2689, 0.7311],[0.2689, 0.7311]]
```
