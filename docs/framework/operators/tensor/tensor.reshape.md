# tensor.reshape

```rust 
   fn reshape(self: @Tensor<T>, target_shape: Span<i32>) -> Tensor<T>;
```

Returns a new tensor with the specified target shape and the same data as the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `target_shape`(Span<i32>) - A span containing the target shape of the tensor.

## Panics

* Panics if the target shape is incompatible with the input tensor's data.

## Returns

A new `Tensor<T>` with the specified target shape and the same data.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reshape_tensor_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reshape` function as follows.
    return tensor.reshape(target_shape: array![2, 4].span());
}
>>> [[0,1,2,3], [4,5,6,7]]
```
