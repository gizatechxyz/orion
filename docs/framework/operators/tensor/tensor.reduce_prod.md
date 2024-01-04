## tensor.reduce_prod

```rust 
   fn reduce_prod(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Reduces a tensor by multiplying its elements along a specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The dimension to reduce.
* `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by multiplying its elements.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reduce_prod_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reduce_prod` function as follows.
    return tensor.reduce_prod(axis: 0, keepdims: false);
}
>>> [[0,5],[12,21]]
```
