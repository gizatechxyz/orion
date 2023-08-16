#tensor.greater_equal

```rust
    fn greater_equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
```

Check if each element of the first tensor is greater than or equal to the corresponding element of the second tensor.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor to be compared
* `other`(`@Tensor<T>`) - The second tensor to be compared

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.

## Examples

Case 1: Compare tensors with same shape

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};

fn greater_equal_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(),
        data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
        extra: Option::None(())
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(),
        data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
        extra: Option::None(())
    );

    // We can call `greater_equal` function as follows.
    return tensor_1.greater_equal(@tensor_2);
}
>>> [1,1,1,1,1,1,0,1,1]
```

Case 2: Compare tensors with different shapes

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};

fn greater_equal_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(),
        data: array![0,1, 2, 3, 4, 5, 6, 7, 8].span(),
        extra: Option::None(())
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3].span(),
        data: array![0, 1, 2].span(),
        extra: Option::None(())
    );

    // We can call `greater_equal` function as follows.
    return tensor_1.greater_equal(@tensor_2);
}
>>> [1,1,1,1,1,1,0,0,0]
```
