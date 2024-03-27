## tensor.reduce_sum

```rust 
   fn reduce_sum(self: @Tensor<T>, axes: Option<Span<i32>>, keepdims: Option<bool>, noop_with_empty_axes: Option<bool>) -> Tensor<T>;
```

Reduces a tensor by summing its elements along a specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axes`(`Option<Span<i32>>`) - Optional input list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, else act as an Identity op when 'noop_with_empty_axes' is true.
* `keepdims`(`Option<bool>`) - Keep the reduced dimension or not, default 1 means keep reduced dimension.
* `noop_with_empty_axes`(`Option<bool>`) - Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. When axes is empty and this attribute is set to true, input tensor will not be reduced,and the output tensor would be equivalent to input tensor.

## Returns

Reduced output tensor.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reduce_sum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reduce_sum` function as follows.
    return tensor.reduce_sum(axes: Option::None, keepdims: false);
}
>>> [[4,6],[8,10]]
```
