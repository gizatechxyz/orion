# tensor.stride

```rust 
   fn stride(self: @Tensor<T>) -> Span<usize>;
```

Computes the stride of each dimension in the tensor.

## Args
* `self`(`@Tensor<T>`) - The input tensor.

## Returns

A span of usize representing the stride for each dimension of the tensor.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn stride_example() -> Span<usize> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `stride` function as follows.
    return tensor.stride();
}
>>> [4,2,1]
```
