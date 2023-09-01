# tensor.at

```rust 
   fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
```

Retrieves the value at the specified indices of a Tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Span<usize>`) - The indices to access element of the Tensor.

## Panics

* Panics if the number of indices provided don't match the number of dimensions in the tensor.

## Returns

The `T` value at the specified indices.

# Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp16x16};


fn at_example() -> u32 {
    let tensor = TensorTrait::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `at` function as follows.
    return tensor.at(indices: array![0, 1, 1].span());
}
>>> 3
```
