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

Let's create new u32 Tensor with constant 42.

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{
    TensorTrait, // we import the trait
    Tensor, // we import the type
    U32Tensor // we import the implementation. 
};

fn constant_of_shape_example() -> Tensor<u32> {
    let tensor = TensorTrait::constant_of_shape(shape: array![3].span(), value: 42);

    return tensor;
}

>>> [42, 42, 42]
```
