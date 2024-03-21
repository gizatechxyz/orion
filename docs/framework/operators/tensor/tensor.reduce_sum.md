## tensor.reduce_sum

```rust 
   fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Computes the sum of the input tensor's elements along the provided axes

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axes`(`Option<Span<usize>>`) - Optional input list of integers, along which to reduce.
* `keepdims`(`Option<bool>`) - If true, retains reduced dimensions with length 1.
* `noop_with_empty_axes`(`Option<bool>`) - Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. When axes is empty and this attribute is set to true, input tensor will not be reduced,and the output tensor would be equivalent to input tensor.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reduce_sum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![3, 2, 2].span(), data: array![1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12].span(),
    );

    // We can call `reduce_sum` function as follows.
    return tensor.reduce_sum(Option::Some(array![1].span()), Option::Some(false), Option::None);
}
>>> [[4, 6] [12, 14] [20, 22]]
```
