# Tensor

A Tensor represents a multi-dimensional array of elements.

```rust
use onnx_cairo::operators::tensor;
```

A `Tensor` represents a multi-dimensional array of elements and is depicted as a struct containing both the tensor's shape and a flattened array of its data. The generic Tensor is defined as follows:

```rust
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}
```

### Data types

ONNX-Cairo supports currently three tensor types.

| Data type                  | dtype               |
| -------------------------- | ------------------- |
| 32-bit integer (signed)    | `Tensor<i32>`       |
| 32-bit integer (unsigned)  | `Tensor<u32>`       |
| Fixed point Q5.26 (signed) | `Tensor<FixedType>` |

***

### Tensor**Trait**

```rust
use onnx_cairo::operators::tensor::core::TensorTrait;
```

`TensorTrait` defines the operations that can be performed on a Tensor.

| function                                           | description                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| [`TensorTrait::new`](tensortrait-new.md)           | Constructs a new Tensor with the given shape and data array.                |
| [`tensor.at`](tensor.at.md)                        | Accesses the element at the given multi-dimensional index.                  |
| [`tensor.min`](tensor.min.md)                      | Returns the minimum value in the tensor.                                    |
| [`tensor.max`](tensor.max.md)                      | Returns the maximum value in the tensor.                                    |
| [`tensor.stride`](tensor.stride.md)                | Computes the stride of each dimension in the tensor.                        |
| [`tensor.ravel_index`](tensor.ravel\_index.md)     | Converts a multi-dimensional index to a one-dimensional index.              |
| [`tensor.unravel_index`](tensor.unravel\_index.md) | Converts a one-dimensional index to a multi-dimensional index.              |
| [`tensor.reshape`](tensor.reshape.md)              | Returns a new tensor with the specified target shape and the same data.     |
| [`tensor.transpose`](tensor.transpose.md)          | Returns a new tensor with the axes rearranged according to the given array. |
| [`tensor.reduce_sum`](tensor.reduce\_sum.md)       | Reduces the tensor by summing along the specified axis.                     |
| [`tensor.argmax`](tensor.argmax.md)                | Returns the index of the maximum value along the specified axis.            |
| [`tensor.matmul`](tensor.matmul.md)                | Performs matrix multiplication.                                             |
| [`tensor.exp`](tensor.exp.md)                      | Calculates the exponential function (e^x) for each element in a tensor.     |

### Arithmetic Operations

`Tensor` implements arithmetic traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`, `-`, `*`, `/` ) Tensors arithmetic operations supports broadcasting.

Two tensors are “broadcastable” if the following rules hold:

* Each tensor has at least one dimension.
* When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

#### Examples

Element-wise add.

```rust
fn element_wise_add_example() -> Tensor<u32> {
    // We instantiate two 3D Tensors here.
    // tensor_1 = [[[0,1],[2,3]],[[4,5],[6,7]]]
    // tensor_2 = [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor_1 = u32_tensor_2x2x2_helper();
    let tensor_2 = u32_tensor_2x2x2_helper();

    // We can add two tensors as follows.
    return tensor_1 + tensor_2;
}
>>> [[[0,2],[4,6]],[[8,10],[12,14]]]
```

Add two tensors of different shapes but compatible in broadcasting.

```rust
fn broadcasting_add_example() -> Tensor<u32> {
    // We instantiate two Tensors here.
    // tensor_1 = [[[0,1],[2,3]],[[4,5],[6,7]]]
    // tensor_2 = [[[ 10],[100]]]
    let tensor_1 = u32_tensor_2x2x2_helper();
    let tensor_2 = u32_tensor_1x2x1_helper();
    
    // We can add two tensors as follows.
    return tensor_1 + tensor_2;
}
>>> [[[10,11],[102,103]],[[14,15],[106,107]]]
```
