# tensor.transpose

```rust 
   fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
```

Returns a new tensor with the axes rearranged according to the given permutation.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axes`(`Span<usize>`) - The usize elements representing the axes to be transposed.

## Panics

* Panics if the length of the axes array is not equal to the rank of the input tensor.

## Returns

A `Tensor<T>` instance with the axes reordered according to the given permutation.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};


fn transpose_tensor_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    // We can call `transpose` function as follows.
    return tensor.transpose(
        axes: array![1, 2, 0].span()
    );
}
>>> [[[0,4],[1,5]],[[2,6],[3,7]]]
```
