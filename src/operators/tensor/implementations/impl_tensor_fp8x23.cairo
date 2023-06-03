//! This module defines and implement a Tensor for FixedType values.

use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor
};
use orion::operators::tensor::math::min::min_fp8x23::min_in_tensor;
use orion::operators::tensor::math::max::max_fp8x23::max_in_tensor;
use orion::operators::tensor::math::equal::equal_fp8x23::equal;
use orion::operators::tensor::math::less::less_fp8x23::less;
use orion::operators::tensor::math::less_equal::less_equal_fp8x23::less_equal;
use orion::operators::tensor::math::abs::abs_fp8x23::abs;
use orion::operators::tensor::math::reduce_sum::reduce_sum_fp8x23::reduce_sum;
use orion::operators::tensor::math::argmax::argmax_fp8x23::argmax;
use orion::operators::tensor::linalg::matmul::matmul_fp8x23::matmul;
use orion::operators::tensor::linalg::transpose::transpose_fp8x23::transpose;
use orion::operators::tensor::math::exp::exp_fp8x23::exp;
use orion::operators::tensor::math::arithmetic::arithmetic_fp8x23::{add, sub, mul, div};
use orion::operators::tensor::math::greater::greater_fp8x23::greater;
use orion::operators::tensor::math::greater_equal::greater_equal_fp8x23::greater_equal;
use orion::utils::check_gas;

impl Tensor_fp8x23 of TensorTrait<FixedType<fp8x23>, fp8x23> {
    fn new(shape: Span<usize>, data: Span<FixedType<fp8x23>>) -> Tensor<FixedType<fp8x23>> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<FixedType<fp8x23>>, indices: Span<usize>) -> FixedType<fp8x23> {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<FixedType<fp8x23>>) -> FixedType<fp8x23> {
        min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<FixedType<fp8x23>>) -> FixedType<fp8x23> {
        max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<FixedType<fp8x23>>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FixedType<fp8x23>>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FixedType<fp8x23>>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(
        self: @Tensor<FixedType<fp8x23>>, target_shape: Span<usize>
    ) -> Tensor<FixedType<fp8x23>> {
        reshape(self, target_shape)
    }

    fn reduce_sum(
        self: @Tensor<FixedType<fp8x23>>, axis: usize, keepdims: bool
    ) -> Tensor<FixedType<fp8x23>> {
        reduce_sum(self, axis, keepdims)
    }

    fn argmax(self: @Tensor<FixedType<fp8x23>>, axis: usize) -> Tensor<usize> {
        argmax(self, axis)
    }

    fn transpose(self: @Tensor<FixedType<fp8x23>>, axes: Span<usize>) -> Tensor<FixedType<fp8x23>> {
        transpose(self, axes)
    }

    fn matmul(
        self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>
    ) -> Tensor<FixedType<fp8x23>> {
        matmul(self, other)
    }

    fn exp(self: @Tensor<FixedType<fp8x23>>) -> Tensor<FixedType<fp8x23>> {
        exp(self)
    }

    fn eq(self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>) -> Tensor<usize> {
        equal(self, other)
    }

    fn greater(
        self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>
    ) -> Tensor<usize> {
        greater(self, other)
    }

    fn greater_equal(
        self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>
    ) -> Tensor<usize> {
        greater_equal(self, other)
    }

    fn less(self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>) -> Tensor<usize> {
        less(self, other)
    }

    fn less_equal(
        self: @Tensor<FixedType<fp8x23>>, other: @Tensor<FixedType<fp8x23>>
    ) -> Tensor<usize> {
        less_equal(self, other)
    }

    fn abs(self: @Tensor<FixedType<fp8x23>>) -> Tensor<FixedType<fp8x23>> {
        abs(self)
    }
}

/// Implements addition for `Tensor<FixedType>` using the `Add` trait.
impl FixedTypeTensorAdd of Add<Tensor<FixedType<fp8x23>>> {
    /// Adds two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise addition.
    fn add(
        lhs: Tensor<FixedType<fp8x23>>, rhs: Tensor<FixedType<fp8x23>>
    ) -> Tensor<FixedType<fp8x23>> {
        add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FixedType>` using the `Sub` trait.
impl FixedTypeTensorSub of Sub<Tensor<FixedType<fp8x23>>> {
    /// Subtracts two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise subtraction.
    fn sub(
        lhs: Tensor<FixedType<fp8x23>>, rhs: Tensor<FixedType<fp8x23>>
    ) -> Tensor<FixedType<fp8x23>> {
        sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FixedType>` using the `Mul` trait.
impl FixedTypeTensorMul of Mul<Tensor<FixedType<fp8x23>>> {
    /// Multiplies two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise multiplication.
    fn mul(
        lhs: Tensor<FixedType<fp8x23>>, rhs: Tensor<FixedType<fp8x23>>
    ) -> Tensor<FixedType<fp8x23>> {
        mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FixedType>` using the `Div` trait.
impl FixedTypeTensorDiv of Div<Tensor<FixedType<fp8x23>>> {
    /// Divides two `Tensor<FixedType>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FixedType>` instance representing the result of the element-wise division.
    fn div(
        lhs: Tensor<FixedType<fp8x23>>, rhs: Tensor<FixedType<fp8x23>>
    ) -> Tensor<FixedType<fp8x23>> {
        div(@lhs, @rhs)
    }
}
