//! This module defines and implement a Tensor for FixedType values.

use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::types::FixedType;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor
};
use orion::operators::tensor::math::min::min_fp::min_in_tensor;
use orion::operators::tensor::math::max::max_fp::max_in_tensor;
use orion::operators::tensor::math::reduce_sum::reduce_sum_fp::reduce_sum;
use orion::operators::tensor::math::argmax::argmax_fp::argmax;
use orion::operators::tensor::linalg::matmul::matmul_fp::matmul;
use orion::operators::tensor::linalg::transpose::transpose_fp::transpose;
use orion::operators::tensor::math::exp::exp_fp::exp;
use orion::operators::tensor::math::arithmetic::arithmetic_fp::{add, sub, mul, div};
use orion::utils::check_gas;

impl FixedTypeTensor of TensorTrait<FixedType> {
    fn new(shape: Span<usize>, data: Span<FixedType>) -> Tensor<FixedType> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<FixedType>, indices: Span<usize>) -> FixedType {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<FixedType>) -> FixedType {
        min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<FixedType>) -> FixedType {
        max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<FixedType>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FixedType>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FixedType>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<FixedType>, target_shape: Span<usize>) -> Tensor<FixedType> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<FixedType>, axis: usize, keepdims: bool) -> Tensor<FixedType> {
        reduce_sum(self, axis, keepdims)
    }

    fn argmax(self: @Tensor<FixedType>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    fn transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Tensor<FixedType> {
        transpose(self, axes)
    }

    fn matmul(self: @Tensor<FixedType>, other: @Tensor<FixedType>) -> Tensor<FixedType> {
        matmul(self, other)
    }

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
