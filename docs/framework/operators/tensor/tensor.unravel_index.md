# tensor.unravel_index

```rust 
   fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
```

Converts a one-dimensional index to a multi-dimensional index.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Span<usize>`) - The index to unravel.

## Panics

* Panics if the index is out of bounds of the Tensor shape.

## Returns

The unraveled indices corresponding to the given index.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn unravel_index_example() -> Span<usize> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `unravel_index` function as follows.
    return tensor.unravel_index(3);
}
>>> [0,1,1] 
// This means that the value of index 3 of Tensor.data
// can be found at indices [0,1,1] in multidimensional representation.
```
