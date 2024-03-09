# tensor.reshape

```rust 
   fn reshape(self: @Tensor<T>, target_shape: Span<usize>, allowzero: Option<usize>) -> Tensor<T>;
```

Returns a new tensor with the specified target shape and the same data as the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `target_shape`(Span<usize>) - A span containing the target shape of the tensor.
* `allowzero`(Option<usize>) - (Optional) By default, when any value in the 'shape' input is equal to zero the corresponding dimension value is copied from the input tensor dynamically. allowzero=1 indicates that if any value in the 'shape' input is set to zero, the zero value is honored.

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
    return tensor.reshape(target_shape: array![2, 4].span(), allowzero: Option::None);
}
>>> [[0,1,2,3], [4,5,6,7]]
```
