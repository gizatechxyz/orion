# NNTrait::hard_swish

```rust 
   fn hard_swish(tensor: @Tensor<T>) -> Tensor<T>;
```

Applies the HardSwish function to an n-dimensional input tensor.


## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

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

fn hard_swish_example() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<FP16x16>::new(
        shape: array![2, 2, 2].span(),
        data: array![
            FixedTrait::new(87989, true),
            FixedTrait::new(13107, false),
            FixedTrait::new(32768, true),
            FixedTrait::new(65536, false),
            FixedTrait::new(89090, true),
            FixedTrait::new(13107, false),
            FixedTrait::new(38988, true),
            FixedTrait::new(78990, false),
        ]
            .span(),
    );

    return NNTrait::hard_swish(@tensor);
}
>>> [[[-0.37089539,  0.10665894],
      [-0.20832825,  0.66665649]],

     [[-0.37171936,  0.10665894],
      [-0.23846436,  0.84474182]]]
```
