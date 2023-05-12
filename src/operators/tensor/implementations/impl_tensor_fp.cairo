//! This module defines and implement a Tensor for FixedType values.

use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor
};
use onnx_cairo::operators::tensor::math::min::min_fp::min_in_tensor;
use onnx_cairo::operators::tensor::math::max::max_fp::max_in_tensor;
use onnx_cairo::operators::tensor::math::reduce_sum::reduce_sum_fp::reduce_sum;
use onnx_cairo::operators::tensor::math::argmax::argmax_fp::argmax;
use onnx_cairo::operators::tensor::linalg::matmul::matmul_fp::matmul;
use onnx_cairo::operators::tensor::linalg::transpose::transpose_fp::transpose;
use onnx_cairo::operators::tensor::math::exp::exp_fp::exp;
use onnx_cairo::operators::tensor::math::arithmetic::arithmetic_fp::{add, sub, mul, div};
use onnx_cairo::utils::check_gas;

impl FixedTypeTensor of TensorTrait<FixedType> {
    /// Creates a new FixedType tensor with the given shape and data.
    ///
    /// # Arguments
    /// * `shape` - A span representing the shape of the tensor.
    /// * `data` -  A span containing the array of FixedType elements.
    ///
    /// # Panics
    /// * Panics if the shape and data length are incompatible.
    ///
    /// # Returns
    /// * A new `Tensor<FixedType>` instance.
    fn new(shape: Span<usize>, data: Span<FixedType>) -> Tensor<FixedType> {
        new_tensor(shape, data)
    }

    /// Retrieves the value at the specified indices of an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `indices` - A span representing the indices to access.
    ///
    /// # Panics
    /// * Panics the number of indices provided don't match the number of dimensions in the tensor.
    ///
    /// # Returns
    /// * The FixedType value at the specified indices.
    fn at(self: @Tensor<FixedType>, indices: Span<usize>) -> FixedType {
        *at_tensor(self, indices)
    }

    /// Finds the minimum value in an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The minimum FixedType value in the tensor.
    fn min(self: @Tensor<FixedType>) -> FixedType {
        min_in_tensor(*self.data)
    }

    /// Finds the maximum value in an FixedType tensor.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * The maximum FixedType value in the tensor.
    fn max(self: @Tensor<FixedType>) -> FixedType {
        max_in_tensor(*self.data)
    }

    /// Computes the stride of an FixedType tensor.
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
    fn stride(self: @Tensor<FixedType>) -> Span<usize> {
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
    fn ravel_index(self: @Tensor<FixedType>, indices: Span<usize>) -> usize {
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
    fn unravel_index(self: @Tensor<FixedType>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    /// Reshapes an FixedType tensor to the target shape.
    ///
    /// # Arguments
    /// * `self` - The input tensor.
    /// * `target_shape` - A span representing the target shape.
    ///
    /// # Panics
    /// * Panics if the target shape is incompatible with the input tensor's data.
    ///
    /// # Returns
    /// * A new `Tensor<FixedType>` instance with the specified shape.
    fn reshape(self: @Tensor<FixedType>, target_shape: Span<usize>) -> Tensor<FixedType> {
        reshape(self, target_shape)
    }

    /// Reduces an FixedType tensor along the given axis by summing its elements.
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
    /// * A new `Tensor<FixedType>` instance with the specified axis reduced by summing its elements.
    fn reduce_sum(self: @Tensor<FixedType>, axis: usize, keepdims: bool) -> Tensor<FixedType> {
        reduce_sum(self, axis, keepdims)
    }

    /// Computes the indices of the maximum values along the given axis of an FixedType tensor.
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
    fn argmax(self: @Tensor<FixedType>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    /// Transposes an FixedType tensor according to the specified axes.
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
    /// * A new `Tensor<FixedType>` instance with the axes transposed according to the specified order.
    fn transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Tensor<FixedType> {
        transpose(self, axes)
    }

    /// Performs matrix multiplication between two FixedType tensors.
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
    /// * A new `Tensor<FixedType>` resulting from the matrix multiplication.
    fn matmul(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
        matmul(self, other)
    }

    /// Calculates the exponential function (e^x) for each element in a tensor of FixedType values.
    ///
    /// # Arguments
    ///
    /// * `self` - A tensor of FixedType values representing the input tensor.
    ///
    /// # Panics
    ///
    /// * If gas limit is reached during computation.
    ///
    /// # Returns
    ///
    /// * A tensor of fixed point numbers representing the result 
    fn exp(self: @Tensor<FixedType>) -> Tensor<FixedType> {
        exp(self)
    }
}

/// Implements addition for `Tensor<FixedType>` using the `Add` trait.
impl FixedTypeTensorAdd of Add<Tensor<FixedType>> {
    /// Adds two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FixedType>, rhs: Tensor<FixedType>) -> Tensor<FixedType> {
        add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FixedType>` using the `Sub` trait.
impl FixedTypeTensorSub of Sub<Tensor<FixedType>> {
    /// Subtracts two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FixedType>, rhs: Tensor<FixedType>) -> Tensor<FixedType> {
        sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FixedType>` using the `Mul` trait.
impl FixedTypeTensorMul of Mul<Tensor<FixedType>> {
    /// Multiplies two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FixedType>, rhs: Tensor<FixedType>) -> Tensor<FixedType> {
        mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FixedType>` using the `Div` trait.
impl FixedTypeTensorDiv of Div<Tensor<FixedType>> {
    /// Divides two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FixedType>, rhs: Tensor<FixedType>) -> Tensor<FixedType> {
        div(@lhs, @rhs)
    }
}
