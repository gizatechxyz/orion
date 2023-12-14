# tensor.min

```rust 
   fn min(tensors: Span<Tensor<T>>) -> Tensor<T>;
```

Returns the element-wise minumum values from a list of input tensors
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors

## Returns 

A new `Tensor<T>` containing the element-wise minimum values

## Panics

* Panics if tensor array is empty
* Panics if the shapes are not equal or broadcastable

## Examples

Case 1: Process tensors with same shape

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn min_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 3, 1, 2].span(),);
    let result = TensorTrait::min(tensors: array![tensor1, tensor2].span());
    return result;
}
>>> [0, 1, 1, 2]

    result.shape
>>> (2, 2)
```

Case 2: Process tensors with different shapes

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn min_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![1, 2].span(), data: array![1, 4].span(),);
    let result = TensorTrait::min(tensors: array![tensor1, tensor2].span());
    return result;
}
>>> [0, 1, 1, 4]

    result.shape
>>> (2, 2)
```
