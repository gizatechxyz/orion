# tensor.gather_elements

```rust 
   fn gather_elements(self: @Tensor<T>, indices: Tensor<i32>, axis: Option<i32>) -> Tensor<T>;
```

GatherElements is an indexing operation that produces its output by indexing into the input data tensor at index positions determined by elements of the indices tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Tensor<i32>`) - Tensor of indices.
* `axis`(`Option<i32>`) - Axis to gather_elements on. Default: axis=0.

## Panics

* Panics if index values are not within bounds [-s, s-1] along axis of size s.

## Returns 

A new `Tensor<T>` .

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn gather_elements_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), 
        data: array![[ 1, 2, 3],[4, 5, 6], [7, 8, 9]].span(), 
    );
    let indices = TensorTrait::<i32>::new(
        shape: array![1, 2, 0].span(), 
        data: array![2, 0, 0].span(), 
    );

    return tensor.gather_elements(
        indices: indices, 
        axis: Option::None(()), 
    );
}
>>> [[4. 8. 3.]
     [7. 2. 3.]]
```
