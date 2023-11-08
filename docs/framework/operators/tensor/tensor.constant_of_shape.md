# tensor.constant_of_shape

```rust 
   fn constant_of_shape(shape: Span<usize>, value: T) -> Tensor<T>;
```

Returns a new tensor with the given shape and constant value.

## Args

* `shape`(`Span<usize>`) - A span representing the shape of the tensor.
* `value` (`T`) - the constant value.

## Returns

A new `Tensor<T>` instance.

## Examples

Let's create new u32 Tensor with constant 0.

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{
    TensorTrait, // we import the trait
    Tensor, // we import the type
    U32Tensor // we import the implementation. 
};

// 1D TENSOR
fn tensor_1D() -> Tensor<u32> {
    let tensor = TensorTrait::new(shape: array![3].span(), value: 0);

    return tensor;
}

// 2D TENSOR
fn tensor_2D() -> Tensor<u32> {
    let tensor = TensorTrait::new(shape: array![2, 2].span(), value: 10);

    return tensor;
}

// 3D TENSOR
fn tensor_3D() -> Tensor<u32> {
    let tensor = TensorTrait::new(
        shape: array![2, 2, 2].span(), value: 20,
    );

    return tensor;
}
```
