# tensor.unique

```rust
    fn unique(self: @Tensor<T>, axis: Option<usize>, sorted: Option<bool>) -> (Tensor<T>, Tensor<i32>, Tensor<i32>, Tensor<i32>);
```

Identifies the unique elements or subtensors of a tensor, with an optional axis parameter for subtensor slicing.
This function returns a tuple containing the tensor of unique elements or subtensors, and optionally,
tensors for indices, inverse indices, and counts of unique elements.
* `axis`(`Option<i32>`) - Specifies the dimension along which to find unique subtensors. A None value means the unique
                          elements of the tensor will be returned in a flattened form. A negative value indicates
                          dimension counting from the end.
* `sorted`(`Option<bool>`) - Determines if the unique elements should be returned in ascending order. Defaults to true.

## Returns

A tuple containing:
* A Tensor<T> with unique values or subtensors from self.
* A Tensor<i32> with the first occurrence indices of unique elements in self. If axis is given, it returns indices
  along that axis; otherwise, it refers to the flattened tensor.
* A Tensor<i32> mapping each element of self to its index in the unique tensor. If axis is specified, it maps to
  the subtensor index; otherwise, it maps to the unique flattened tensor.
* A Tensor<i32> for the counts of each unique element or subtensor in self.


## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn unique_flat_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![1, 6].span(), 
        data: array![[2, 1, 1, 3, 4, 3]].span(), 
    );

    return tensor.unique(
        axis: Option::None(())
        sorted: Option::Some(false) 
    );
}
>>> (
        [2, 1, 3, 4],
        [0, 1, 3, 4],
        [0, 1, 1, 2, 3, 2],
        [1, 2, 2, 1]
    )
```

or

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn unique_axis_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), 
        data: array![[ 1, 0, 0],
                     [ 1, 0, 0],
                     [ 2, 3, 4]].span(), 
    );

    return tensor.unique(
        axis: Option::Some(0)
        sorted: Option::Some(true) 
    );
}
>>> (    
        [[ 1, 0, 0],
         [ 2, 3, 4]],
        [0, 2],
        [0, 0, 1],
        [2, 1]
    )
```
