# Tensor

A Tensor represents a multi-dimensional array of elements.

```rust
use orion::operators::tensor;
```

A `Tensor` represents a multi-dimensional array of elements and is depicted as a struct containing both the tensor's shape,a flattened array of its data and extra parameters. The generic Tensor is defined as follows:

```rust
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
    extra: Option<ExtraParams>
}
```

`ExtraParams` is a struct containing additional parameters for the tensor.

```rust
struct ExtraParams {
    fixed_point: Option<FixedImpl>
}
```

**ExtraParams**
| Params      | dtype               | default     | desciption                                               |
| ----------- | ------------------- | ----------- | -------------------------------------------------------- |
| fixed_point | `Option<FixedImpl>` | `FP16x16()` | Specifies the type of Fixed Point a Tensor can supports. |

### Data types

Orion supports currently these tensor types.

| Data type                 | dtype               |
| ------------------------- | ------------------- |
| 32-bit integer (signed)   | `Tensor<i32>`       |
| 8-bit integer (signed)   | `Tensor<i8>`        |
| 32-bit integer (unsigned) | `Tensor<u32>`       |
| Fixed point  (signed)     | `Tensor<FixedType>` |

---

### Tensor**Trait**

```rust
use orion::operators::tensor::core::TensorTrait;
```

`TensorTrait` defines the operations that can be performed on a Tensor.

| function | description |
| --- | --- |
| [`tensor.new`](tensor.new.md) | Constructs a new Tensor with the given shape and data array. |
| [`tensor.at`](tensor.at.md) | Accesses the element at the given multi-dimensional index. |
| [`tensor.min`](tensor.min.md) | Returns the minimum value in the tensor.     |
| [`tensor.max`](tensor.max.md) | Returns the maximum value in the tensor. |
| [`tensor.stride`](tensor.stride.md) | Computes the stride of each dimension in the tensor. |
| [`tensor.ravel_index`](tensor.ravel\_index.md) | Converts a multi-dimensional index to a one-dimensional index. |
| [`tensor.unravel_index`](tensor.unravel\_index.md) | Converts a one-dimensional index to a multi-dimensional index. |
| [`tensor.reshape`](tensor.reshape.md) | Returns a new tensor with the specified target shape and the same data.  |
| [`tensor.transpose`](tensor.transpose.md) | Returns a new tensor with the axes rearranged according to the given array. |
| [`tensor.reduce_sum`](tensor.reduce\_sum.md) | Reduces the tensor by summing along the specified axis. |
| [`tensor.argmax`](tensor.argmax.md) | Returns the index of the maximum value along the specified axis.   |
| [`tensor.argmin`](tensor.argmin.md) | Returns the index of the minimum value along the specified axis. |
| [`tensor.matmul`](tensor.matmul.md) | Performs matrix multiplication.  |
| [`tensor.exp`](tensor.exp.md) | Calculates the exponential function (e^x) for each element in a tensor. |
| [`tensor.ln`](tensor.ln.md) | Computes the natural log of all elements of the input tensor. |
| [`tensor.eq`](tensor.eq.md) | Check if two tensors are equal element-wise. |
| [`tensor.greater`](tensor.greater.md) | Check if each element of the first tensor is greater than the corresponding element of the second tensor. |
| [`tensor.greater_equal`](tensor.greater\_equal.md) | Check if each element of the first tensor is greater than or equal to the corresponding element of the second tensor. |
| [`tensor.less`](tensor.less.md) | Check if each element of the first tensor is less than the corresponding element of the second tensor. |
| [`tensor.less_equal`](tensor.less\_equal.md) | Check if each element of the first tensor is less than or equal to the corresponding element of the second tensor. |
| [`tensor.abs`](tensor.abs.md) | Computes the absolute value of all elements in the input tensor. |
| [`tensor.ceil`](tensor.ceil.md) | Rounds up the value of each element in the input tensor. |
| [`tensor.cumsum`](tensor.cumsum.md) | Returns the cumulative sum of the elements along a given axis. |
| [`tensor.sin`](tensor.sin.md) | Computes the sine value of each element in the input tensor. |
| [`tensor.cos`](tensor.cos.md) | Computes the cosine value of each element in the input tensor. |
| [`tensor.asin`](tensor.asin.md) | Returns the arcsine (inverse of sine) value of each element in the input tensor. |
| [`tensor.flatten`](tensor.flatten.md) | Flattens the input tensor into a 2D tensor. |
| [`tensor.acosh`](tensor.acosh.md) | Computes the inverse hyperbolic cosine of all elements of the input tensor. |
| [`tensor.asinh`](tensor.asinh.md) | Computes the inverse hyperbolic sine of all elements of the input tensor. |
| [`tensor.cosh`](tensor.cosh.md) | Computes the hyperbolic cosine of all elements of the input tensor. |
| [`tensor.tanh`](tensor.tanh.md) | Computes the hyperbolic tangent of all elements of the input tensor. |
| [`tensor.sinh`](tensor.sinh.md) | Computes the hyperbolic sine of all elements of the input tensor. |
| [`tensor.atan`](tensor.atan.md) | Computes the arctangent (inverse of tangent) of the input tensor. |

### Arithmetic Operations

`Tensor` implements arithmetic traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`, `-`, `*`, `/` ) Tensors arithmetic operations supports broadcasting.

Two tensors are “broadcastable” if the following rules hold:

- Each tensor has at least one dimension.
- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

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
