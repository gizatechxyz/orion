## tensor.sequence_construct

```rust 
   fn sequence_construct(tensors: Array<Tensor<T>>) -> Array<Tensor<T>>;
```

Constructs a tensor sequence containing the input tensors.

## Args

* `tensors`(`Array<Tensor<T>>`) - The array of input tensors.

## Panics 

* Panics if input tensor array is empty.

## Returns

A tensor sequence `Array<Tensor<T>>` containing the input tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn sequence_construct_example() -> Array<Tensor<usize>> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    let result = TensorTrait::sequence_construct(tensors: array![tensor1, tensor2]);
    return result;
}
>>> [[0, 1, 2, 3], [4, 5, 6, 7]]
```
