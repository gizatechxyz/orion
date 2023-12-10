# tensor.onehot

```rust 
   fn onehot(self: @Tensor<T>, depth: usize, axis: Option<usize>, values: Span<usize>) -> Tensor<usize>;
```

Produces one-hot tensor based on input.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `depth`(`usize`) - Scalar or Rank 1 tensor containing exactly one element, specifying the number of classes in one-hot tensor.
* `axis`(`Option<bool>`) - Axis along which one-hot representation in added. Default: axis=-1.
* `values`(`Span<usize>`) - Rank 1 tensor containing exactly two elements, in the format [off_value, on_value]   

## Panics

* Panics if values is not equal to 2.

## Returns 

A new `Tensor<T>` one-hot encode of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn onehot_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2,2].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
            FixedTrait::new_unscaled(3, false),
        ]
            .span(),
    );    

    return tensor.onehot(depth: 3, axis: Option::None(()), values: array![0, 1].span());
}
>>> [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
```
