//! This module defines and implement a Tensor for FP8x23 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor, tensor_eq
};
use orion::operators::tensor::math;
use orion::operators::tensor::implementations::tensor_u32_fp8x23::Tensor_u32_fp8x23;

impl Tensor_fp8x23 of TensorTrait<FP8x23, FP8x23> {
    fn new(shape: Span<usize>, data: Span<FP8x23>, extra: Option<ExtraParams>) -> Tensor<FP8x23> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<FP8x23>, indices: Span<usize>) -> FP8x23 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<FP8x23>) -> FP8x23 {
        math::min::min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<FP8x23>) -> FP8x23 {
        math::max::max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<FP8x23>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FP8x23>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FP8x23>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<FP8x23>, target_shape: Span<usize>) -> Tensor<FP8x23> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<FP8x23>, axis: usize, keepdims: bool) -> Tensor<FP8x23> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<FP8x23>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax::<FP8x23, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<FP8x23>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin::<FP8x23, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<FP8x23>, axes: Span<usize>) -> Tensor<FP8x23> {
        //transpose(self, axes)
        panic(array![])
    }

    fn matmul(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        //matmul(self, other)
        panic(array![])
    }

    fn exp(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::exp::exp_from_fp::<FP8x23, u32>(*self)
    }

    fn log(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::log::log_from_fp::<FP8x23, u32>(*self)
    }

    fn equal(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        //greater_equal(self, other)
        panic(array![])
    }

    fn less(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        //less(self, other)
        panic(array![])
    }

    fn less_equal(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        //less_equal(self, other)
        panic(array![])
    }

    fn abs(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        //math::abs::abs(*self)
        panic(array![])
    }

    fn ceil(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        panic(array!['not supported with FP8x23'])
    }

    fn sin(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // sin_FP8x23(self).unwrap()
        panic(array![])
    }

    fn cos(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // cos_FP8x23(self).unwrap()
        panic(array![])
    }

    fn asin(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // panic(array!['not supported with FP8x23'])
        panic(array![])
    }

    fn cumsum(
        self: @Tensor<FP8x23>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP8x23> {
        // cumsum(self, axis, exclusive, reverse)
        panic(array![])
    }

    fn flatten(self: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        //flatten(self, axis)
        panic(array![])
    }

    fn sinh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // sinh_FP8x23(self).unwrap()
        panic(array![])
    }

    fn tanh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        //  tanh_FP8x23(self).unwrap()
        panic(array![])
    }

    fn cosh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // cosh_FP8x23(self).unwrap()
        panic(array![])
    }

    fn acosh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // acosh_FP8x23(self).unwrap()
        panic(array![])
    }

    fn asinh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // asinh_FP8x23(self).unwrap()
        panic(array![])
    }

    fn atan(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // atan_FP8x23(self).unwrap()
        panic(array![])
    }

    fn xor(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        // xor(self, other)
        panic(array![])
    }

    fn or(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        // or(self, other)
        panic(array![])
    }
    fn acos(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        panic(array!['not supported with FP8x23'])
    }

    fn onehot(
        self: @Tensor<FP8x23>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP8x23> {
        // onehot(self, depth, axis, values)
        panic(array![])
    }

    fn sqrt(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // sqrt_FP8x23(self).unwrap()
        panic(array![])
    }

    fn concat(tensors: Span<Tensor<FP8x23>>, axis: usize,) -> Tensor<FP8x23> {
        // concat_FP8x23(tensors, axis)
        panic(array![])
    }
}

/// Implements addition for `Tensor<FP8x23>` using the `Add` trait.
impl FP8x23TensorAdd of Add<Tensor<FP8x23>> {
    /// Adds two `Tensor<FP8x23>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FP8x23>, rhs: Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FP8x23>` using the `Sub` trait.
impl FP8x23TensorSub of Sub<Tensor<FP8x23>> {
    /// Subtracts two `Tensor<FP8x23>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FP8x23>, rhs: Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FP8x23>` using the `Mul` trait.
impl FP8x23TensorMul of Mul<Tensor<FP8x23>> {
    /// Multiplies two `Tensor<FP8x23>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FP8x23>, rhs: Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FP8x23>` using the `Div` trait.
impl FP8x23TensorDiv of Div<Tensor<FP8x23>> {
    /// Divides two `Tensor<FP8x23>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FP8x23>, rhs: Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<FP8x23>` using the `PartialEq` trait.
impl FP8x23TensorPartialEq of PartialEq<Tensor<FP8x23>> {
    fn eq(lhs: @Tensor<FP8x23>, rhs: @Tensor<FP8x23>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<FP8x23>, rhs: @Tensor<FP8x23>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}
