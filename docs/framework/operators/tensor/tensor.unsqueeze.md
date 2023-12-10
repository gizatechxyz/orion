# tensor.unsqueeze

```rust 
   fn unsqueeze(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
```

Insert single-dimensional entries to the shape of an input tensor (data). Takes one required input axes -
which contains a list of dimension indices and this operator will insert a dimension of value 1 into the
corresponding index of the output tensor (expanded).

## Args

* `self`(`@Tensor<T>`) - Tensor of data to unsquezee.
* `axes`(`Span<usize>`) - List of integers indicating the dimensions to be inserted. 

## Panics

* Panics if the given axes have duplicate elements.
* Panics if one of the given axes is invalid.

## Returns 

Reshaped `Tensor<T>` with same data as input.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn unsqueeze_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 4].span(), 
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    );

    return tensor.unsqueeze(
        axes: array![0, 3].span(), 
    );
}
>>> [[[[0]
       [1]
       [2]
       [3]]

      [[4]
       [5]
       [6]
       [7]]]]
```
