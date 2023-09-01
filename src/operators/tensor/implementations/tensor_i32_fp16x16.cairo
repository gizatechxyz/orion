//! This module defines and implement a Tensor for i32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{Into, TryInto};

use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor,
    tensor_eq
};
use orion::operators::tensor::{math, linalg, quantization};
use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::operators::tensor::implementations::tensor_i8_fp16x16::{
    Tensor_i8_fp16x16, TensorI8IntoTensorI32
};
use orion::numbers::i8;

impl Tensor_i32_fp16x16 of TensorTrait<i32, FP16x16> {
    fn new(shape: Span<usize>, data: Span<i32>,) -> Tensor<i32> {
        new_tensor(shape, data)
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
        math::argmax::argmax::<i32, FP16x16, u32>(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin::<i32, FP16x16, u32>(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::exp::exp_from_int::<i32, FP16x16, u32>(*self)
    }

    fn log(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::log::log_from_int::<i32, FP16x16, u32>(*self)
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

    fn sin(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::sin::sin_from_int(*self)
    }

    fn cos(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::cos::cos_from_int(*self)
    }

    fn asin(self: @Tensor<i32>) -> Tensor<FP16x16> {
        panic(array!['not supported with i32'])
    }

    fn cumsum(
        self: @Tensor<i32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i32> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::sinh::sinh_from_int(*self)
    }

    fn tanh(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::tanh::tanh_from_int(*self)
    }

    fn cosh(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::cosh::cosh_from_int(*self)
    }

    fn acosh(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::acosh::acosh_from_int(*self)
    }

    fn asinh(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::asinh::asinh_from_int(*self)
    }

    fn atan(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::atan::atan_from_int(*self)
    }

    fn xor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::or::or(self, other)
    }
    fn acos(self: @Tensor<i32>) -> Tensor<FP16x16> {
        panic(array!['not supported with i32'])
    }

    fn onehot(
        self: @Tensor<i32>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i32> {
        math::onehot::onehot_from_int(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<i32>) -> Tensor<FP16x16> {
        math::sqrt::sqrt_from_int(*self)
    }

    fn concat(tensors: Span<Tensor<i32>>, axis: usize,) -> Tensor<i32> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
    ) -> Tensor::<i8> {
        quantization::quantize_linear::quantize_linear(
            self,
            y_scale,
            y_zero_point,
            i32 { mag: 128, sign: true },
            i32 { mag: 127, sign: false },
        )
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
    ) -> Tensor::<i32> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
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

impl U32TryIntoU32 of TryInto<u32, u32> {
    fn try_into(self: u32) -> Option<u32> {
        Option::Some(self)
    }
}
