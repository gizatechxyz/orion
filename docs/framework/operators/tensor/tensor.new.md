# tensor.new

```rust 
   fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
```

Returns a new tensor with the given shape and data.

## Args

* `shape`(`Span<usize>`) - A span representing the shape of the tensor.
* `data` (`Span<T>`) - A span containing the array of elements.

## Panics

* Panics if the shape and data length are incompatible.

## Returns

A new `Tensor<T>` instance.

## Examples

Let's create new u32 Tensors.

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{
    TensorTrait, // we import the trait
    Tensor, // we import the type
    U32Tensor // we import the implementation. 
};

// 1D TENSOR
fn tensor_1D() -> Tensor<u32> {
    let tensor = TensorTrait::new(shape: array![3].span(), data: array![0, 1, 2].span());

    return tensor;
}

// 2D TENSOR
fn tensor_2D() -> Tensor<u32> {
    let tensor = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());

    return tensor;
}

// 3D TENSOR
fn tensor_3D() -> Tensor<u32> {
    let tensor = TensorTrait::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    return tensor;
}
```
