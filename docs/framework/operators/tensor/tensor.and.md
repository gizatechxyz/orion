#tensor.and

```rust
    fn and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool>;
```

Computes the logical AND of two tensors element-wise.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<bool>`) - The first tensor to be compared
* `other`(`@Tensor<bool>`) - The second tensor to be compared

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<bool>` with the same shape as the broadcasted inputs.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn and_example() -> Tensor<bool> {
    let tensor_1 = TensorTrait::<bool>::new(
        shape: array![3, 3].span(), data: array![false, true, false, false, false, true, true, false, true, false, false, true].span(),
    );

    let tensor_2 = TensorTrait::<bool>::new(
        shape: array![3, 3].span(), data: array![false, false, true, true, false, true, false, true, false, true, false, true].span(),
    );

    return tensor_1.and(@tensor_2);
}
>>> [false, false, false, false, false, true, false, false, false, false, false, true]
```
