#tensor.bitwise_xor

```rust
    fn bitwise_xor(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
```

Computes the bitwise XOR of two tensors element-wise.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor to be compared
* `other`(`@Tensor<T>`) - The second tensor to be compared

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<T>` with the same shape as the broadcasted inputs.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn xor_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 0, 4, 5, 0, 6, 2].span(),
    );

    return tensor_1.bitwise_xor(@tensor_2);
}
>>> [0,0,0,3,0,0,6,1,10]
```
