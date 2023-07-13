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
// 1D TENSOR
fn tensor_1D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(3);
		
    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);

    let extra = Option::<ExtraParams>::None(());
		
    let tensor = TensorTrait::<u32>::new(shape.span(), data.span(), extra);
		
    return tensor;
}

// 2D TENSOR
fn tensor_2D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(shape.span(), data.span(), extra);

    return tensor;
}

// 3D TENSOR
fn tensor_3D() -> Tensor<u32> {
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(shape.span(), data.span(), extra);

    return tensor;
}
```
