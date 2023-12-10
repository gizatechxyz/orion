# tensor.gather

```rust 
   fn gather(self: @Tensor<T>, indices: Tensor<T>, axis: Option<usize>) -> Tensor<T>;
```

Gather entries of the axis dimension of data.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Tensor<T>`) - Tensor of indices.
* `axis`(`Option<usize>`) - Axis to gather on. Default: axis=0.

## Panics

* Panics if index values are not within bounds [-s, s-1] along axis of size s.

## Returns 

A new `Tensor<T>` .

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn gather_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 3].span(), 
        data: array![[ 1, 2, 3],[4, 5, 6]].span(), 
    );
    let indices = TensorTrait::<u32>::new(
        shape: array![1, 1].span(), 
        data: array![1, 0].span(), 
    );

    return tensor.gather(
        indices: indices, 
        axis: Option::None(()), 
    );
}
>>> [[4. 5. 6.]
     [1. 2. 3.]]
```
