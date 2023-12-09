# tensor.slice

```rust 
   fn slice(self: @Tensor<T>, starts: Span<usize>, ends: Span<usize>, axes: Option<Span<usize>>, steps: Option<Span<usize>>) -> Tensor<usize>;
```

Produces a slice of the input tensor along multiple axes.

## Args

* `self`(`@Tensor<T>`) - Tensor of data to extract slices from.
* `starts`(Span<usize>) - 1-D tensor of starting indices of corresponding axis in `axes`
* `ends`(Span<usize>) - 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
* `axes`(Option<Span<usize>>) - 1-D tensor of axes that `starts` and `ends` apply to. 
* `steps`(Option<Span<usize>>) - 1-D tensor of slice step of corresponding axis in `axes`.    

## Panics

* Panics if the length of starts is not equal to the length of ends.
* Panics if the length of starts is not equal to the length of axes.
* Panics if the length of starts is not equal to the length of steps.

## Returns 

A new `Tensor<T>` slice of the input tensor.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn slice_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 4].span(), 
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    );

    return tensor.slice(
        starts: array![0, 2].span(), 
        ends: array![2, 4].span(), 
        axis: Option::None(()), 
        steps: Option::Some(array![1, 1].span())
    );
}
>>> [[2 3]
     [6 7]]
```
