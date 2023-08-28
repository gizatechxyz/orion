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

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

fn concat_example() -> Tensor<FixedType> {
    let tensor1 = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), 
        data: array![0, 1, 2, 3].span(), 
        extra: Option::None(())
    );

    let tensor2 = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), 
        data: array![0, 1, 2, 3].span(), 
        extra: Option::None(())
    );

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
