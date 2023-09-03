# tensor.concat

```rust 
   fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
```

Concatenate a list of tensors into a single tensor.

## Args

* `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors.
* `axis`(`usize`) -  Axis to concat on.

## Panics

* Panic if tensor length is not greater than 1.
* Panics if dimension is not greater than axis.

## Returns 

A new `Tensor<T>` concatenated tensor of the input tensors.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn concat_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 0);
    return result;
}
>>> [[0. 1.]
     [2. 3.],
     [0. 1.]
     [2. 3.]]

    result.shape
>>> (4, 2)

   let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 1);
   return result;
}
>>> [[0. 1., 0., 1.]
     [2. 3., 2., 3.]]

    result.shape
>>> (2, 4 ) 
```
