# tensor.ravel_index

```rust 
    fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
```

Converts a multi-dimensional index to a one-dimensional index.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Span<usize>`) - The indices of the Tensor to ravel.

## Panics 

* Panics if the indices are out of bounds of the Tensor shape.

## Returns

The index corresponding to the given indices.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};


fn ravel_index_example() -> usize {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    // We can call `ravel_index` function as follows.
    return tensor.ravel_index(
        indices: array![1, 3, 0].span()
    );
}
>>> 10 
// This means that the value of indices [1,3,0] 
// of a multidimensional array can be found at index 10 of Tensor.data.
```
   