//! This module defines and implement a Tensor for u32 values.

use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor
};
use onnx_cairo::operators::tensor::math::min::min_u32::min_in_tensor;
use onnx_cairo::operators::tensor::math::max::max_u32::max_in_tensor;
use onnx_cairo::operators::tensor::math::reduce_sum::reduce_sum_u32::reduce_sum;
use onnx_cairo::operators::tensor::math::argmax::argmax_u32::argmax;
use onnx_cairo::operators::tensor::linalg::matmul::matmul_u32::matmul;
use onnx_cairo::operators::tensor::linalg::transpose::transpose_u32::transpose;
use onnx_cairo::operators::tensor::math::exp::exp_u32::exp;
use onnx_cairo::operators::tensor::math::arithmetic::arithmetic_u32::{add, sub, mul, div};
use onnx_cairo::utils::check_gas;

impl U32Tensor of TensorTrait<u32> {
    /// Creates a new u32 tensor with the given shape and data.
    ///
    /// # Arguments
    /// * `shape` - A span representing the shape of the tensor.
    /// * `data` -  A span containing the data array of u32 elements.
    ///
    /// # Panics
    /// * Panics if the shape and data length are incompatible.
    ///
    /// # Returns
    /// * A new `Tensor<u32>` instance.
    fn new(shape: Span<usize>, data: Span<u32>) -> Tensor<u32> {
        new_tensor(shape, data)
    }

    /// Retrieves the value at the specified indices of an u32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices to access.
    ///
    /// # Panics
    /// * Panics the number of indices provided don't match the number of dimensions in the tensor.
    ///
    /// # Returns
    /// * The u32 value at the specified indices.
    fn at(self: @Tensor<u32>, indices: Span<usize>) -> u32 {
        *at_tensor(self, indices)
    }

    /// Finds the minimum value in an u32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The minimum u32 value in the tensor.
    fn min(self: @Tensor<u32>) -> u32 {
        min_in_tensor(*self.data)
    }

    /// Finds the maximum value in an u32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The maximum u32 value in the tensor.
    fn max(self: @Tensor<u32>) -> u32 {
        max_in_tensor(*self.data)
    }

    /// Computes the stride of an u32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if shape is empty.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A span representing the stride of the tensor.
    fn stride(self: @Tensor<u32>) -> Span<usize> {
        stride(*self.shape)
    }

    /// Converts a multi-dimensional index to a one-dimensional index.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices.
    ///
    /// # Panics
    /// * Panics if the indices are out of bounds for the given shape.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The raveled index corresponding to the given indices.
    fn ravel_index(self: @Tensor<u32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    /// Converts a one-dimensional index to a multi-dimensional index.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `index` - The index to unravel.
    ///
    /// # Panics
    /// * Panics if the index is out of bounds for the given shape.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A span representing the unraveled indices corresponding to the given index.
    fn unravel_index(self: @Tensor<u32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    /// Reshapes an u32 tensor to the target shape.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `target_shape` - A span representing the target shape.
    ///
    /// # Panics
    /// * Panics if the target shape is incompatible with the input tensor's data.
    ///
    /// # Returns
    /// * A new `Tensor<u32>` instance with the specified shape.
    fn reshape(self: @Tensor<u32>, target_shape: Span<usize>) -> Tensor<u32> {
        reshape(self, target_shape)
    }

    /// Reduces an u32 tensor along the given axis by summing its elements.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axis` - The axis along which to reduce the tensor.
    ///
    /// # Panics
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<u32>` instance with the specified axis reduced by summing its elements.
    fn reduce_sum(self: @Tensor<u32>, axis: usize, keepdims: bool) -> Tensor<u32> {
        reduce_sum(self, axis, keepdims)
    }

    /// Computes the indices of the maximum values along the given axis of an u32 tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axis` - The axis along which to compute the argmax.
    ///
    /// # Panics
    /// * Panics if axis is not in the range of the input tensor's dimensions.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<usize>` instance containing the indices of the maximum values along the specified axis.
    fn argmax(self: @Tensor<u32>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    /// Transposes an u32 tensor according to the specified axes.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `axes` -  A span containing the array representing the order in which the axes should be transposed.
    ///
    /// # Panics
    /// * Panics if the length of the axes array is not equal to the rank of the input tensor.
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new `Tensor<u32>` instance with the axes transposed according to the specified order.
    fn transpose(self: @Tensor<u32>, axes: Span<usize>) -> Tensor<u32> {
        transpose(self, axes)
    }

    /// Performs matrix multiplication between two u32 tensors.
    ///
    /// # Arguments
    /// * `self` - The first tensor.
    /// * `other` - The second tensor.
    ///
    /// # Behavior
    /// The behavior depends on the dimensionality of the tensors as follows:
    /// * If both tensors are 1-dimensional, the dot product is returned.
    /// * If both arguments are 2-dimensional, the matrix-matrix product is returned.
    /// * If the first argument is 1-dimensional and the second argument is 2-dimensional,
    ///   a 1 is prepended to its dimension for the purpose of the matrix multiply. After
    ///   the matrix multiply, the prepended dimension is removed.
    /// * If the first argument is 2-dimensional and the second argument is 1-dimensional,
    ///   the matrix-vector product is returned.
    ///
    /// # Panics
    /// * Panics if the dimension of the tensors is higher than two.
    ///
    /// # Returns
    /// * A new `Tensor<u32>` resulting from the matrix multiplication.
    fn matmul(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
        matmul(self, other)
    }

    /// Calculates the exponential function (e^x) for each element in a tensor of u32 values.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor of u32 values representing the input tensor.
    ///
    /// # Panics
    ///
    /// * If gas limit is reached during computation.
    ///
    /// # Returns
    ///
    /// * A tensor of fixed point numbers representing the result 
    fn exp(self: @Tensor<u32>) -> Tensor<FixedType> {
        exp(self)
    }
}

/// Implements addition for `Tensor<u32>` using the `Add` trait.
impl U32TensorAdd of Add<Tensor<u32>> {
    /// Adds two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<u32>` using the `Sub` trait.
impl U32TensorSub of Sub<Tensor<u32>> {
    /// Subtracts two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<u32>` using the `Mul` trait.
impl U32TensorMul of Mul<Tensor<u32>> {
    /// Multiplies two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<u32>` using the `Div` trait.
impl U32TensorDiv of Div<Tensor<u32>> {
    /// Divides two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        div(@lhs, @rhs)
    }
}
