## tensor.reduce_l1

```rust 
   fn reduce_l1(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Computes the L1 norm of the input tensor's elements along the provided axes.

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
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reduce_l1_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reduce_l1` function as follows.
    return tensor.reduce_l1(axis: 1, keepdims: false);
}
>>> [[2,4],[10,12]]
```
