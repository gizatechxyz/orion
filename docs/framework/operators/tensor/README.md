# Tensor

A Tensor represents a multi-dimensional array of elements.

A `Tensor` represents a multi-dimensional array of elements and is depicted as a struct containing both the tensor's shape and a flattened array of its data. The generic Tensor is defined as follows:

```rust
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>
}
```

### Data types

Orion supports currently these tensor types.

| Data type                 | dtype                                             |
| ------------------------- | ------------------------------------------------- |
| 32-bit integer (signed)   | `Tensor<i32>`                                     |
| 8-bit integer (signed)    | `Tensor<i8>`                                      |
| 32-bit integer (unsigned) | `Tensor<u32>`                                     |
| Fixed point (signed)      | `Tensor<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |

***

### Tensor**Trait**

```rust
use orion::operators::tensor::TensorTrait;
```

`TensorTrait` defines the operations that can be performed on a Tensor.

| function | description |
| --- | --- |
| [`tensor.new`](tensor.new.md) | Returns a new tensor with the given shape and data. |
| [`tensor.reshape`](tensor.reshape.md) | Returns a new tensor with the specified target shape and the same data as the input tensor. |
| [`tensor.flatten`](tensor.flatten.md) | Flattens the input tensor into a 2D tensor. |
| [`tensor.constant_of_shape`](tensor.constant\_of\_shape.md) | Generate a tensor with given value and shape. |
| [`tensor.transpose`](tensor.transpose.md) | Returns a new tensor with the axes rearranged according to the given permutation. |
| [`tensor.at`](tensor.at.md) | Retrieves the value at the specified indices of a Tensor. |
| [`tensor.ravel_index`](tensor.ravel\_index.md) | Converts a multi-dimensional index to a one-dimensional index. |
| [`tensor.unravel_index`](tensor.unravel\_index.md) | Converts a one-dimensional index to a multi-dimensional index. |
| [`tensor.equal`](tensor.equal.md) | Check if two tensors are equal element-wise. |
| [`tensor.greater`](tensor.greater.md) | Check if each element of the first tensor is greater than the corresponding element of the second tensor. |
| [`tensor.greater_equal`](tensor.greater\_equal.md) | Check if each element of the first tensor is greater than or equal to the corresponding element of the second tensor. |
| [`tensor.less`](tensor.less.md) | Check if each element of the first tensor is less than the corresponding element of the second tensor. |
| [`tensor.less_equal`](tensor.less\_equal.md) | Check if each element of the first tensor is less than or equal to the corresponding element of the second tensor. |
| [`tensor.or`](tensor.or.md) | Computes the logical OR of two tensors element-wise. |
| [`tensor.xor`](tensor.xor.md) | Computes the logical XOR of two tensors element-wise. |
| [`tensor.stride`](tensor.stride.md) | Computes the stride of each dimension in the tensor. |
| [`tensor.onehot`](tensor.onehot.md) | Produces one-hot tensor based on input. |
| [`tensor.max_in_tensor`](tensor.max\_in\_tensor.md) | Returns the maximum value in the tensor. |
| [`tensor.min_in_tensor`](tensor.min\_in\_tensor.md) | Returns the minimum value in the tensor. |
| [`tensor.min`](tensor.min.md) | Returns the minimum value in the tensor. |
| [`tensor.max`](tensor.max.md) | Returns the maximum value in the tensor. |
| [`tensor.reduce_sum`](tensor.reduce\_sum.md) | Reduces a tensor by summing its elements along a specified axis. |
| [`tensor.reduce_prod`](tensor.reduce\_prod.md) | Reduces a tensor to its products along specified axis. |
| [`tensor.argmax`](tensor.argmax.md) | Returns the index of the maximum value along the specified axis. |
| [`tensor.argmin`](tensor.argmin.md) | Returns the index of the minimum value along the specified axis. |
| [`tensor.cumsum`](tensor.cumsum.md) | Performs cumulative sum of the input elements along the given axis. |
| [`tensor.matmul`](tensor.matmul.md) | Performs matrix product of two tensors. |
| [`tensor.exp`](tensor.exp.md) | Computes the exponential of all elements of the input tensor. |
| [`tensor.log`](tensor.log.md) | Computes the natural log of all elements of the input tensor. |
| [`tensor.abs`](tensor.abs.md) | Computes the absolute value of all elements in the input tensor. |
| [`tensor.neg`](tensor.neg.md) | Computes the negation of all elements in the input tensor. |
| [`tensor.ceil`](tensor.ceil.md) | Rounds up the value of each element in the input tensor. |
| [`tensor.sqrt`](tensor.sqrt.md) | Computes the square root of all elements of the input tensor. |
| [`tensor.sin`](tensor.sin.md) | Computes the sine of all elements of the input tensor. |
| [`tensor.cos`](tensor.cos.md) | Computes the cosine of all elements of the input tensor. |
| [`tensor.atan`](tensor.atan.md) | Computes the arctangent (inverse of tangent) of all elements of the input tensor. |
| [`tensor.asin`](tensor.asin.md) | Computes the arcsine (inverse of sine) of all elements of the input tensor. |
| [`tensor.acos`](tensor.acos.md) | Computes the arccosine (inverse of cosine) of all elements of the input tensor. |
| [`tensor.sinh`](tensor.sinh.md) | Computes the hyperbolic sine of all elements of the input tensor. |
| [`tensor.tanh`](tensor.tanh.md) | Computes the hyperbolic tangent of all elements of the input tensor. |
| [`tensor.cosh`](tensor.cosh.md) | Computes the hyperbolic cosine of all elements of the input tensor. |
| [`tensor.asinh`](tensor.asinh.md) | Computes the inverse hyperbolic sine of all elements of the input tensor. |
| [`tensor.acosh`](tensor.acosh.md) | Computes the inverse hyperbolic cosine of all elements of the input tensor. |
| [`tensor.slice`](tensor.slice.md) | Produces a slice of the input tensor along multiple axes.  |
| [`tensor.concat`](tensor.concat.md) | Concatenate a list of tensors into a single tensor. |
| [`tensor.quantize_linear`](tensor.quantize\_linear.md) | Quantizes a Tensor to i8 using linear quantization. |
| [`tensor.dequantize_linear`](tensor.dequantize\_linear.md) | Dequantizes an i8 Tensor using linear dequantization. |
| [`tensor.qlinear_add`](tensor.qlinear\_add.md) | Performs the sum of two quantized i8 Tensors. |
| [`tensor.qlinear_mul`](tensor.qlinear\_mul.md) | Performs the element-wise multiplication of quantized Tensors. |
| [`tensor.qlinear_matmul`](tensor.qlinear\_matmul.md) | Performs the product of two quantized i8 Tensors. |
| [`tensor.qlinear_concat`](tensor.qlinear\_concat.md) | Concatenate a list of tensors after dequantizing them with their respective scales and zero_points and returns the quantized result. |
| [`tensor.qlinear_leakyrelu`](tensor.qlinear\_leakyrelu.md) | Applies the Leaky Relu operator to a quantized Tensor |
| [`tensor.gather`](tensor.gather.md) | Gather entries of the axis dimension of data. |
| [`tensor.nonzero`](tensor.nonzero.md) | Produces indices of the elements that are non-zero (in row-major order - by dimension). |
| [`tensor.squeeze`](tensor.squeeze.md) | Removes dimensions of size 1 from the shape of a tensor. |
| [`tensor.unsqueeze`](tensor.unsqueeze.md) | Inserts single-dimensional entries to the shape of an input tensor. |
| [`tensor.sign`](tensor.sign.md) | Calculates the sign of the given input tensor element-wise. |
| [`tensor.clip`](tensor.clip.md) | Clip operator limits the given input within an interval. |
| [`tensor.and`](tensor.and.md) | Computes the logical AND of two tensors element-wise.  |
| [`tensor.identity`](tensor.identity.md) | Return a Tensor with the same shape and contents as input. |
| [`tensor.where`](tensor.where.md) | Return elements chosen from x or y depending on condition. |
| [`tensor.bitwise_and`](tensor.bitwise\_and.md) | Computes the bitwise AND of two tensors element-wise. |
| [`tensor.bitwise_xor`](tensor.bitwise\_xor.md) | Computes the bitwise XOR of two tensors element-wise. |
| [`tensor.bitwise_or`](tensor.bitwise\_or.md) | Computes the bitwise OR of two tensors element-wise. |
| [`tensor.resize`](tensor.resize.md) | Resizes the input tensor. |
| [`tensor.round`](tensor.round.md) | Computes the round value of all elements in the input tensor. |
| [`tensor.reduce_l1`](tensor.reduce\_l1.md) | Computes the L1 norm of the input tensor's elements along the provided axes. |
| [`tensor.trilu`](tensor.trilu.md) | Returns the upper or lower triangular part of a tensor or a batch of 2D matrices. |
| [`tensor.scatter`](tensor.scatter.md) | Produces a copy of input data, and updates value to values specified by updates at specific index positions specified by indices. |
| [`tensor.reduce_sum_square`](tensor.reduce\_sum\_square.md) | Computes the sum square of the input tensor's elements along the provided axes.  |
| [`tensor.reduce_l2`](tensor.reduce\_l2.md) | Computes the L2 norm of the input tensor's elements along the provided axes. |
| [`tensor.gather_elements`](tensor.gather\_elements.md) | GatherElements is an indexing operation that produces its output by indexing into the input data tensor at index positions determined by elements of the indices tensor. |
| [`tensor.reduce_min`](tensor.reduce\_min.md) | Computes the min of the input tensor's elements along the provided axes. |
| [`tensor.reduce_mean`](tensor.reduce\_mean.md) | Computes the mean of the input tensor's elements along the provided axes. |
| [`tensor.pow`](tensor.pow.md) | Pow takes input data (Tensor) and exponent Tensor, and produces one output data (Tensor) where the function f(x) = x^exponent, is applied to the data tensor elementwise. |
| [`tensor.binarizer`](tensor.binarizer.md) | Maps the values of a tensor element-wise to 0 or 1 based on the comparison against a threshold value. |
| [`tensor.array_feature_extractor`](tensor.array\_feature\_extractor.md) | Selects elements of the input tensor based on the indices passed applied to the last tensor axis. |
| [`tensor.reduce_min`](tensor.reduce\_min.md) | Computes the min of the input tensor's elements along the provided axes. |
| [`tensor.is_nan`](tensor.is\_nan.md) | Returns which elements of the input are NaN. |
| [`tensor.is_inf`](tensor.is\_inf.md) | Maps infinity to true and other values to false. |
| [`tensor.not`](tensor.not.md) | Computes the logical negation of all elements in the input tensor. |
| [`tensor.gather_nd`](tensor.gather\_nd.md) | Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers slices of data into an output tensor of rank q + r - indices_shape[-1] - 1 - b. |
| [`tensor.reduce_log_sum`](tensor.reduce\_log\_sum.md) | Computes the log sum of the input tensor's elements along the provided axes.  |
| [`tensor.erf`](tensor.erf.md) | Computes the error function of the given input tensor element-wise. |
| [`tensor.reduce_log_sum_exp`](tensor.reduce\_log\_sum\_exp.md) | Computes the log sum of the exponentials of the input tensor's elements along the provided axes. |
| [`tensor.layer_normalization`](tensor.layer\_normalization.md) | computes the layer normalization of the input tensor. |
| [`tensor.split`](tensor.split.md) | Split a tensor into a list of tensors, along the specified ‘axis’.  |
| [`tensor.random_uniform_like`](tensor.random\_uniform\_like.md) | RandomUniformLike generates a tensor with random values using a uniform distribution, matching the shape of the input tensor. |
| [`tensor.split_to_sequence`](tensor.split\_to\_sequence.md) | Split a tensor into a sequence of tensors, along the specified ‘axis’. |
| [`tensor.range`](tensor.range.md) | Generate a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive). |
| [`tensor.hann_window`](tensor.hann\_window.md) | Generates a Hann window as described in the paper https://ieeexplore.ieee.org/document/1455106. |
| [`tensor.hamming_window`](tensor.hamming\_window.md) | Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106. |
| [`tensor.blackman_window`](tensor.blackman\_window.md) | Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106. |
| [`tensor.reverse_sequence`](tensor.reverse\_sequence.md) | Reverse batch of sequences having different lengths specified by sequence_lens. |
| [`tensor.optional`](tensor.optional.md) | Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute, or a non-empty value containing the input element. |
| [`tensor.dynamic_quantize_linear`](tensor.dynamic\_quantize\_linear.md) | Computes the Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.  |
| [`tensor.scatter_nd`](tensor.scatter\_nd.md) | The output of the operation is produced by creating a copy of the input data, and then updating its value to values specified by updates at specific index positions specified by indices. Its output shape is the same as the shape of data |
| [`tensor.label_encoder`](tensor.label\_encoder.md) | Maps each element in the input tensor to another value. |
| [`tensor.mean`](tensor.mean.md) | Element-wise mean of each of the input tensors. |

## Arithmetic Operations

`Tensor` implements arithmetic traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`, `-`, `*`, `/` ). Tensors arithmetic operations supports broadcasting.

Two tensors are “broadcastable” if the following rules hold:

* Each tensor has at least one dimension.
* When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

#### Examples

Element-wise add.

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, U32TensorAdd};

fn element_wise_add_example() -> Tensor<u32> {
    // We instantiate two 3D Tensors here.
    let tensor_1 = TensorTrait::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );
    let tensor_2 = TensorTrait::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can add two tensors as follows.
    return tensor_1 + tensor_2;
}
>>> [[[0,2],[4,6]],[[8,10],[12,14]]]
```

Add two tensors of different shapes but compatible in broadcasting.

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, U32TensorAdd};

fn broadcasting_add_example() -> Tensor<u32> {
    // We instantiate two 3D Tensors here.
    let tensor_1 = TensorTrait::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );
    let tensor_2 = TensorTrait::new(
        shape: array![1, 2, 1].span(),
        data: array![10, 100].span(),
    );

    // We can add two tensors as follows.
    return tensor_1 + tensor_2;
}
>>> [[[10,11],[102,103]],[[14,15],[106,107]]]
```
