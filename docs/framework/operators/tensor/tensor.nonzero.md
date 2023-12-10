# tensor.nonzero

```rust 
   fn nonzero(self: @Tensor<T>) -> Tensor<usize>;
```

Produces indices of the elements that are non-zero (in row-major order - by dimension).

## Args

* `self`(`@Tensor<T>`) - Tensor of data to calculate non-zero indices.  

## Returns 

A new `Tensor<usize>` indices of the elements that are non-zero (in row-major order - by dimension).

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn nonzero_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 4].span(), 
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
    );

    return tensor.nonzero();
}
>>> [[0 0 0 1 1 1 1]
     [1 2 3 0 1 2 3]]
```
