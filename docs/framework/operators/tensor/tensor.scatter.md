# tensor.scatter

```rust 
   fn scatter(self: @Tensor<T>, updates: Tensor<T>, indices: Tensor<usize>,  axis: Option<usize>, reduction: Option<usize>) -> Tensor<T>;
```

Produces a copy of input data, and updates value to values specified by updates at specific index positions specified by indices.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `updates`(`Tensor<T>`) - The inupdates tensor.
* `indices`(`Tensor<T>`) - Tensor of indices.
* `axis`(`Option<usize>`) - Axis to scatter on. Default: axis=0.
* `reduction`(`Option<usize>`) - Reduction operation. Default: reduction='none'.

## Panics

* Panics if index values are not within bounds [-s, s-1] along axis of size s.

## Returns 

A new `Tensor<T>` .

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn scatter_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![3, 5].span(), 
        data: array![[ 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0]].span(), 
    );
    let updates = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), 
        data: array![[ 1, 2, 3],
                     [ 4, 5, 6],
                     [ 7, 8, 9]].span(), 
    );
    let indices = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), 
        data: array![[ 0, 1, 2],
                     [ 2, 0, 1],
                     [ 1, 0, 1]].span(), 
    );

    return tensor.scatter(
        updates: updates
        indices: indices, 
        axis: Option::None(()), 
        reduction: Option::None(()), 
    );
}
>>> [[ 1, 8, 0, 0, 0],
     [ 7, 2, 9, 0, 0],
     [ 4, 0, 3, 0, 0]]
```
