# tensor.new

```rust 
   fn new(shape: Span<usize>, data: Span<T>, extra: Option<ExtraParams>) -> Tensor<T>;
```

Returns a new tensor with the given shape and data.

## Args

* `shape`(`Span<usize>`) - A span representing the shape of the tensor.
* `data` (`Span<T>`) - A span containing the array of elements.
* `extra` (`Option<ExtraParams>`) - A parameter for extra tensor options.

## Panics

* Panics if the shape and data length are incompatible.

## Returns

A new `Tensor<T>` instance.

## Examples

Let's create new u32 Tensors.

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};

// 1D TENSOR
fn tensor_1D() -> Tensor<u32> {
    let tensor = TensorTrait::new(
        shape: array![3].span(),
        data: array![0, 1, 2].span(),
        extra: Option::None(())
    );

    return tensor;
}

// 2D TENSOR
fn tensor_2D() -> Tensor<u32> {
    let tensor = TensorTrait::new(
        shape: array![2, 2].span(), 
        data: array![0, 1, 2, 3].span(), 
        extra: Option::None(())
    );

    return tensor;
}

// 3D TENSOR
fn tensor_3D() -> Tensor<u32> {
    let tensor = TensorTrait::new(
        shape: array![2, 2, 2].span(), 
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(), 
        extra: Option::None(())
    );

    return tensor;
}
```
