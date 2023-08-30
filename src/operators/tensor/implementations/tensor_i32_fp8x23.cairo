//! This module defines and implement a Tensor for i32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{Into, TryInto};

use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::{FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor, tensor_eq
};
use orion::operators::tensor::math;
use orion::operators::tensor::implementations::tensor_u32_fp8x23::Tensor_u32_fp8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;

impl Tensor_i32_fp8x23 of TensorTrait<i32, FP8x23> {
    fn new(shape: Span<usize>, data: Span<i32>, extra: Option<ExtraParams>) -> Tensor<i32> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<i32>) -> i32 {
        math::min::min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<i32>) -> i32 {
        math::max::max_in_tensor(*self.data)
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
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax::<i32, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin::<i32, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        //transpose(self, axes)
        panic(array![])
    }

    fn matmul(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
        //matmul(self, other)
        panic(array![])
    }

    fn exp(self: @Tensor<i32>) -> Tensor<FP8x23> {
        math::exp::exp_from_int::<i32, FP8x23, u32>(*self)
    }

    fn log(self: @Tensor<i32>) -> Tensor<FP8x23> {
        math::log::log_from_int::<i32, FP8x23, u32>(*self)
    }

    fn equal(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<i32>) -> Tensor<i32> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported with i32'])
    }

    fn sin(self: @Tensor<i32>) -> Tensor<FP8x23> {
        math::sin::sin_from_int(*self)
    }

    fn cos(self: @Tensor<i32>) -> Tensor<FP8x23> {
        math::cos::cos_from_int(*self)
    }

    fn asin(self: @Tensor<i32>) -> Tensor<FP8x23> {
        panic(array!['not supported with i32'])
    }

    fn cumsum(
        self: @Tensor<i32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i32> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        //flatten(self, axis)
        panic(array![])
    }

    fn sinh(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // sinh_i32(self).unwrap()
        panic(array![])
    }

    fn tanh(self: @Tensor<i32>) -> Tensor<FP8x23> {
        //  tanh_i32(self).unwrap()
        panic(array![])
    }

    fn cosh(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // cosh_i32(self).unwrap()
        panic(array![])
    }

    fn acosh(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // acosh_i32(self).unwrap()
        panic(array![])
    }

    fn asinh(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // asinh_i32(self).unwrap()
        panic(array![])
    }

    fn atan(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // atan_i32(self).unwrap()
        panic(array![])
    }

    fn xor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        // xor(self, other)
        panic(array![])
    }

    fn or(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        // or(self, other)
        panic(array![])
    }
    fn acos(self: @Tensor<i32>) -> Tensor<FP8x23> {
        panic(array!['not supported with i32'])
    }

    fn onehot(
        self: @Tensor<i32>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i32> {
        // onehot(self, depth, axis, values)
        panic(array![])
    }

    fn sqrt(self: @Tensor<i32>) -> Tensor<FP8x23> {
        // sqrt_i32(self).unwrap()
        panic(array![])
    }

    fn concat(tensors: Span<Tensor<i32>>, axis: usize,) -> Tensor<i32> {
        // concat_i32(tensors, axis)
        panic(array![])
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
        math::arithmetic::add(@lhs, @rhs)
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
        math::arithmetic::sub(@lhs, @rhs)
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
        math::arithmetic::mul(@lhs, @rhs)
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
        math::arithmetic::div(@lhs, @rhs)
    }
}


/// Implements partial equal for two `Tensor<i32>` using the `PartialEq` trait.
impl i32TensorPartialEq of PartialEq<Tensor<i32>> {
    fn eq(lhs: @Tensor<i32>, rhs: @Tensor<i32>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<i32>, rhs: @Tensor<i32>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}
