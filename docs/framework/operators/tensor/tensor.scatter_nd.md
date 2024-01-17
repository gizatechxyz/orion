# tensor.scatter_nd

```rust 
   fn scatter_nd(self: @Tensor<T>, updates: Tensor<T>, indices: Tensor<usize>,  reduction: Option<usize>) -> Tensor<T>;
```

Produces a copy of input data, and updates value to values specified by updates at specific index positions specified by indices.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `updates`(`Tensor<T>`) - The updates tensor.
* `indices`(`Tensor<T>`) - Tensor of indices.
* `reduction`(`Option<usize>`) - Reduction operation. Default: reduction='none'.

## Panics

* Panics if index values are not within bounds [-s, s-1] along axis of size s.
* Panics if indices last axis is greater than data rank.

## Returns 

A new `Tensor<T>` .

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn scatter_nd_example() -> Tensor<u32> {
   let tensor = TensorTrait::<u32>::new(
       shape: array![4, 4, 4].span(), 
       data: array![1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
            7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
            5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8].span()
   );

   let updates = TensorTrait::<u32>::new(
       shape: array![2, 4, 4].span(), 
       data: array![5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 1, 1, 1, 1, 2, 2,
                   2, 2, 3, 3, 3, 3, 4, 4, 4, 4].span(), 
   );

   let indices = TensorTrait::<u32>::new(
       shape: array![2, 1].span(), 
       data: array![0, 2].span(), 
   );

    return tensor.scatter_nd(
        updates: updates
        indices: indices, 
        reduction: Option::Some('add'), 
    );
}
>>> [[[ 6.,  7.,  8.,  9.],
       [11., 12., 13., 14.],
       [15., 14., 13., 12.],
       [12., 11., 10.,  9.]],

   [[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 8.,  7.,  6.,  5.],
       [ 4.,  3.,  2.,  1.]],

   [[ 9.,  8.,  7.,  6.],
       [ 6.,  5.,  4.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 9., 10., 11., 12.]],

   [[ 8.,  7.,  6.,  5.],
       [ 4.,  3.,  2.,  1.],
       [ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.]]]
```
