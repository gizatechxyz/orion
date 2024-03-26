## tensor.reduce_max

```rust 
   fn reduce_max(self: @Tensor<T>, axes: Option<Span<usize>>, keepdims: Option<bool>, noop_with_empty_axes: Option<bool>) -> Tensor<T>;
```

Computes the max of the input tensor's elements along the provided axes.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axes`(`Option<Span<usize>>`) - Optional input list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, else act as an Identity op when 'noop_with_empty_axes' is true.
* `keepdims`(`Option<bool>`) - Keep the reduced dimension or not, default true means keep reduced dimension.
* `noop_with_empty_axes`(`Option<bool>`) - Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. When axes is empty and this attribute is set to true, input tensor will not be reduced,and the output tensor would be equivalent to input tensor.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axes reduced by maximum of its elements.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reduce_max_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reduce_max` function as follows.
    return tensor.reduce_max(axes: array![1].span(), 
        keepdims: Option::None(()), 
        noop_with_empty_axes:  Option::None(()));
}
>>> [[2,3],[6,7]]
```