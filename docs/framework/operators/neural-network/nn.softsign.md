# NNTrait::softsign

```rust 
   fn softsign(tensor: @Tensor<T>) -> Tensor<T>;
```

Applies the Softsign function to an n-dimensional input Tensor such that the elements of the n-dimensional output Tensor lie in the range \[-1,1]. 

$$
\text{softsign}(x_i) = \frac{x_i}{1 + |x_i|}
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
use orion::operators::nn::{NNTrait, FP8x23NN};
use orion::numbers::{FP8x23, FixedTrait};

fn softsign_example() -> Tensor<FP8x23> {
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

    return NNTrait::softsign(@tensor);
}
>>> [[0,4194304],[5592405,6291456]]
    // The fixed point representation of
    // [[0, 0.5],[0.67, 0.75]]
```
