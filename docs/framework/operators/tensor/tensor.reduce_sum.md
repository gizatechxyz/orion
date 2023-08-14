## tensor.reduce_sum

```rust 
   fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Reduces a tensor by summing its elements along a specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The dimension to reduce.
* `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};


fn reduce_sum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    // We can call `reduce_sum` function as follows.
    return tensor.reduce_sum(
        axis: 0,
        keepdims: false
    );
}
>>> [[4,6],[8,10]]
```
