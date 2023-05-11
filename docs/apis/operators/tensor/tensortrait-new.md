# TensorTrait::new

Returns a new tensor with the given shape and data.

```rust
fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
```

#### Args

| Name    | Type          | Description                                  |
| ------- | ------------- | -------------------------------------------- |
| `shape` | `Span<usize>` | A span representing the shape of the tensor. |
| `data`  | `Span<T>`     | A span containing the array of elements.     |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                             |
| ----------------------------------------------------- |
| Panics if the shape and data length are incompatible. |

#### Returns

A new `Tensor<T>` instance.

> _`<T>` generic type depends on Tensor dtype._

#### Examples

Let’s create new u32 Tensors.

```rust
/// 1D TENSOR
fn tensor_1D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(3);
		
    let mut data = ArrayTrait::new();
    data.append(0_u32);
    data.append(1_u32);
    data.append(2_u32);
		
    let tensor = TensorTrait::<u32>::new(shape.span(), data.span());
		
    return tensor;
}

/// 2D TENSOR
fn tensor_2D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_u32);
    data.append(1_u32);
    data.append(2_u32);
    data.append(3_u32);

    let tensor = TensorTrait::<u32>::new(shape.span(), data.span());

    return tensor;
}

/// 3D TENSOR
fn tensor_3D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_u32);
    data.append(1_u32);
    data.append(2_u32);
    data.append(3_u32);
    data.append(4_u32);
    data.append(5_u32);
    data.append(6_u32);
    data.append(7_u32);

    let tensor = TensorTrait::<u32>::new(shape.span(), data.span());

    return tensor;
}
```
