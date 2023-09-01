# NNTrait::sigmoid

```rust 
   fn sigmoid(tensor: @Tensor<T>) -> Tensor<F>;
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

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_i32_fp8x23};
use orion::operators::nn::{NNTrait, NN_i32_fp8x23};
use orion::numbers::{i32, FP8x23, IntegerTrait};

fn sigmoid_example() -> Tensor<FP8x23> {
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

    return NNTrait::sigmoid(@tensor);
}
>>> [[4194304,6132564],[7388661,7990771]]
    // The fixed point representation of
    // [[0.5, 0.7310586],[0.88079703, 0.95257413]]
```
