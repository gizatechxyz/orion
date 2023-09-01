# NNTrait::softsign

```rust 
   fn softsign(tensor: @Tensor<T>) -> Tensor<F>;
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

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_i32_fp8x23};
use orion::operators::nn::{NNTrait, NN_i32_fp8x23};
use orion::numbers::{i32, FP8x23, IntegerTrait};

fn softsign_example() -> Tensor<FP8x23> {
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

    return NNTrait::softsign(@tensor);
}
>>> [[0,4194304],[5592405,6291456]]
    // The fixed point representation of
    // [[0, 0.5],[0.67, 0.75]]
```
