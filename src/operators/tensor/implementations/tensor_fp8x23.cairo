//! This module defines and implement a Tensor for FP8x23 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::{FixedImpl, FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization};
use orion::operators::tensor::implementations::tensor_u32_fp8x23::Tensor_u32_fp8x23;
use orion::operators::tensor::implementations::tensor_i8_fp8x23::Tensor_i8_fp8x23;
use orion::numbers::i8;

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
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        linalg::matmul::matmul(self, other)
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
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::ceil::ceil(*self)
    }

    fn sin(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::sin::sin_from_fp(*self)
    }

    fn cos(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::cos::cos_from_fp(*self)
    }

    fn asin(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<FP8x23>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP8x23> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::sinh::sinh_from_fp(*self)
    }

    fn tanh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::tanh::tanh_from_fp(*self)
    }

    fn cosh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::cosh::cosh_from_fp(*self)
    }

    fn acosh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::acosh::acosh_from_fp(*self)
    }

    fn asinh(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::asinh::asinh_from_fp(*self)
    }

    fn atan(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::atan::atan_from_fp(*self)
    }

    fn xor(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<FP8x23>, other: @Tensor<FP8x23>) -> Tensor<usize> {
        math::or::or(self, other)
    }
    fn acos(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<FP8x23>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP8x23> {
        math::onehot::onehot_from_fp(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        math::sqrt::sqrt_from_fp(*self)
    }

    fn concat(tensors: Span<Tensor<FP8x23>>, axis: usize,) -> Tensor<FP8x23> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<FP8x23>, y_scale: @Tensor<FP8x23>, y_zero_point: @Tensor<FP8x23>
    ) -> Tensor::<i8> {
        quantization::quantize_linear::quantize_linear(
            self,
            y_scale,
            y_zero_point,
            FixedTrait::new(1073741824, true),
            FixedTrait::new(1065353216, false)
        )
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

impl U32TryIntoU32 of TryInto<u32, u32> {
    fn try_into(self: u32) -> Option<u32> {
        Option::Some(self)
    }
}

// Internals

const PRECISION: u32 = 75497; // 0.009

fn relative_eq(lhs: @FP8x23, rhs: @FP8x23) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs.mag != 0 {
        (diff / *lhs).mag
    } else {
        diff.mag
    };

    rel_diff <= PRECISION
}

fn tensor_eq(mut lhs: Tensor<FP8x23>, mut rhs: Tensor<FP8x23>,) -> bool {
    let mut is_eq = true;

    loop {
        if lhs.shape.len() == 0 || !is_eq {
            break;
        }

        is_eq = lhs.shape.pop_front().unwrap() == rhs.shape.pop_front().unwrap();
    };

    if !is_eq {
        return false;
    }

    loop {
        if lhs.data.len() == 0 || !is_eq {
            break;
        }

        is_eq = relative_eq(lhs.data.pop_front().unwrap(), rhs.data.pop_front().unwrap());
    };

    return is_eq;
}
