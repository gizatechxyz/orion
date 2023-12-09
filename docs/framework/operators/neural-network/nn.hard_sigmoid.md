# NNTrait::hard_sigmoid

```rust 
   fn hard_sigmoid(tensor: @Tensor<T>, alpha: @T, beta: @T) -> Tensor<T>;
```

Applies the HardSigmoid function to an n-dimensional input tensor.

$$
\text{HardSigmoid}(x_i) = \text{max}(0, \text{min}(alpha * x + beta, 1))
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.
* `alpha`(`@T`) - value of alpha.
* `beta`(`@T`) - value of beta.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
use orion::operators::nn::{NNTrait, FP8x23NN};
use orion::numbers::{FP16x16, FixedTrait};

fn hard_sigmoid_example() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<FP16x16>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new(0, false),
            FixedTrait::new(13107, false),
            FixedTrait::new(32768, false),
            FixedTrait::new(65536, false),
        ]
            .span(),
    );
    let alpha = FixedTrait::new(13107, false);
    let beta = FixedTrait::new(32768, false);

    return NNTrait::hard_sigmoid(@tensor, @alpha, @beta);
}
>>> [[32768, 35389],[39321, 45875]]
```
