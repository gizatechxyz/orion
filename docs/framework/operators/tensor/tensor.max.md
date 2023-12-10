# tensor.max

```rust 
   fn max(tensors: Span<Tensor<T>>) -> Tensor<T>;
```

Returns the element-wise maximum values from a list of input tensors
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors

## Returns 

A new `Tensor<T>` containing the element-wise maximum values

## Panics

* Panics if tensor array is empty
* Panics if the shapes are not equal or broadcastable

## Examples

Case 1: Process tensors with same shape

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn max_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 3, 1, 2].span(),);
    let result = TensorTrait::max(tensors: array![tensor1, tensor2].span());
    return result;
}
>>> [0, 3, 2, 3]

    result.shape
>>> (2, 2)
```

Case 2: Process tensors with different shapes

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn max_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![1, 2].span(), data: array![1, 4].span(),);
    let result = TensorTrait::max(tensors: array![tensor1, tensor2].span());
    return result;
}
>>> [1, 4, 2, 4]

    result.shape
>>> (2, 2)
```
