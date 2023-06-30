//! This module defines and implement a Tensor for i32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor
};
use orion::operators::tensor::math::min::min_i32::min_in_tensor;
use orion::operators::tensor::math::max::max_i32::max_in_tensor;
use orion::operators::tensor::math::reduce_sum::reduce_sum_i32::reduce_sum;
use orion::operators::tensor::math::argmax::argmax_i32::argmax;
use orion::operators::tensor::math::argmin::argmin_i32::argmin;
use orion::operators::tensor::math::equal::equal_i32::equal;
use orion::operators::tensor::math::greater::greater_i32::greater;
use orion::operators::tensor::math::greater_equal::greater_equal_i32::greater_equal;
use orion::operators::tensor::math::less::less_i32::less;
use orion::operators::tensor::math::less_equal::less_equal_i32::less_equal;
use orion::operators::tensor::math::abs::abs_i32::abs;
use orion::operators::tensor::math::ceil::ceil_i32::ceil;
use orion::operators::tensor::linalg::matmul::matmul_i32::matmul;
use orion::operators::tensor::linalg::transpose::transpose_i32::transpose;
use orion::operators::tensor::math::exp::exp_i32::core::exp_i32;
use orion::operators::tensor::math::ln::ln_i32::core::ln_i32;
use orion::operators::tensor::math::arithmetic::arithmetic_i32::{add, sub, mul, div};
use orion::operators::tensor::math::cumsum::cumsum_i32::cumsum;
use orion::operators::tensor::math::flatten::flatten_i32::flatten;
use orion::operators::tensor::math::sinh::sinh_i32::core::sinh_i32;
use orion::operators::tensor::math::tanh::tanh_i32::core::tanh_i32;
use orion::operators::tensor::math::cosh::cosh_i32::core::cosh_i32;
use orion::utils::check_gas;

use orion::operators::tensor::math::sin::sin_i32::core::sin_i32;

impl Tensor_i32 of TensorTrait<i32> {
    fn new(shape: Span<usize>, data: Span<i32>, extra: Option<ExtraParams>) -> Tensor<i32> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<i32>) -> i32 {
        min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<i32>) -> i32 {
        max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<i32>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<i32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<i32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<i32>, target_shape: Span<usize>) -> Tensor<i32> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<i32>, axis: usize, keepdims: bool) -> Tensor<i32> {
        reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        transpose(self, axes)
    }

    fn matmul(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
        matmul(self, other)
    }

    fn exp(self: @Tensor<i32>) -> Tensor<FixedType> {
        exp_i32(self).unwrap()
    }

    fn ln(self: @Tensor<i32>) -> Tensor<FixedType> {
        ln_i32(self).unwrap()
    }

    fn eq(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        equal(self, other)
    }

    fn greater(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        greater(self, other)
    }

    fn greater_equal(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        greater_equal(self, other)
    }

    fn less(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        less(self, other)
    }

    fn less_equal(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        less_equal(self, other)
    }

    fn abs(self: @Tensor<i32>) -> Tensor<i32> {
        abs(self)
    }

    fn ceil(self: @Tensor<i32>) -> Tensor<i32> {
        ceil(self)
    }

    fn sin(self: @Tensor<i32>) -> Tensor<FixedType> {
        sin_i32(self).unwrap()
    }
    
    fn cumsum(
        self: @Tensor<i32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i32> {
        cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        flatten(self, axis)
    }

    fn sinh(self: @Tensor<i32>) -> Tensor<FixedType> {
        sinh_i32(self).unwrap()
    }

    fn tanh(self: @Tensor<i32>) -> Tensor<FixedType> {
        tanh_i32(self).unwrap()
    }

    fn cosh(self: @Tensor<i32>) -> Tensor<FixedType> {
        cosh_i32(self).unwrap()
    }
}

/// Implements addition for `Tensor<i32>` using the `Add` trait.
impl i32TensorAdd of Add<Tensor<i32>> {
    /// Adds two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<i32>, rhs: Tensor<i32>) -> Tensor<i32> {
        add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<i32>` using the `Sub` trait.
impl i32TensorSub of Sub<Tensor<i32>> {
    /// Subtracts two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<i32>, rhs: Tensor<i32>) -> Tensor<i32> {
        sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<i32>` using the `Mul` trait.
impl i32TensorMul of Mul<Tensor<i32>> {
    /// Multiplies two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<i32>, rhs: Tensor<i32>) -> Tensor<i32> {
        mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<i32>` using the `Div` trait.
impl i32TensorDiv of Div<Tensor<i32>> {
    /// Divides two `Tensor<i32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i32>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<i32>, rhs: Tensor<i32>) -> Tensor<i32> {
        div(@lhs, @rhs)
    }
}
