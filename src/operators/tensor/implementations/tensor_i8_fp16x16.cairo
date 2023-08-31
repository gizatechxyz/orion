//! This module defines and implement a Tensor for i8 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{Into, TryInto};

use orion::numbers::signed_integer::{i8::i8, i32::i32};
use orion::numbers::signed_integer::i8::{i8_to_fp16x16};
use orion::numbers::fixed_point::core::{FixedImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor, tensor_eq
};
use orion::operators::tensor::{math, linalg};
use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;
use orion::operators::tensor::implementations::tensor_i32_fp16x16::Tensor_i32_fp16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;

impl Tensor_i8_fp16x16 of TensorTrait<i8, FP16x16> {
    fn new(shape: Span<usize>, data: Span<i8>, extra: Option<ExtraParams>) -> Tensor<i8> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<i8>, indices: Span<usize>) -> i8 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<i8>) -> i8 {
        math::min::min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<i8>) -> i8 {
        math::max::max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<i8>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<i8>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<i8>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<i8>, target_shape: Span<usize>) -> Tensor<i8> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<i8>, axis: usize, keepdims: bool) -> Tensor<i8> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<i8>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax::<i8, FP16x16, u8>(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i8>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin::<i8, FP16x16, u8>(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i8>, axes: Span<usize>) -> Tensor<i8> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<i8> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::exp::exp_from_int::<i8, FP16x16, u8>(*self)
    }

    fn log(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::log::log_from_int::<i8, FP16x16, u8>(*self)
    }

    fn equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<i8>) -> Tensor<i8> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported with i8'])
    }

    fn sin(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::sin::sin_from_int(*self)
    }

    fn cos(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::cos::cos_from_int(*self)
    }

    fn asin(self: @Tensor<i8>) -> Tensor<FP16x16> {
        panic(array!['not supported with i8'])
    }

    fn cumsum(
        self: @Tensor<i8>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i8> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::sinh::sinh_from_int(*self)
    }

    fn tanh(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::tanh::tanh_from_int(*self)
    }

    fn cosh(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::cosh::cosh_from_int(*self)
    }

    fn acosh(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::acosh::acosh_from_int(*self)
    }

    fn asinh(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::asinh::asinh_from_int(*self)
    }

    fn atan(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::atan::atan_from_int(*self)
    }

    fn xor(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::or::or(self, other)
    }
    fn acos(self: @Tensor<i8>) -> Tensor<FP16x16> {
        panic(array!['not supported with i8'])
    }

    fn onehot(
        self: @Tensor<i8>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i8> {
        math::onehot::onehot_from_int(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<i8>) -> Tensor<FP16x16> {
        math::sqrt::sqrt_from_int(*self)
    }

    fn concat(tensors: Span<Tensor<i8>>, axis: usize,) -> Tensor<i8> {
        math::concat::concat(tensors, axis)
    }
}

/// Implements addition for `Tensor<i8>` using the `Add` trait.
impl i8TensorAdd of Add<Tensor<i8>> {
    /// Adds two `Tensor<i8>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i8>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<i8>, rhs: Tensor<i8>) -> Tensor<i8> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<i8>` using the `Sub` trait.
impl i8TensorSub of Sub<Tensor<i8>> {
    /// Subtracts two `Tensor<i8>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i8>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<i8>, rhs: Tensor<i8>) -> Tensor<i8> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<i8>` using the `Mul` trait.
impl i8TensorMul of Mul<Tensor<i8>> {
    /// Multiplies two `Tensor<i8>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i8>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<i8>, rhs: Tensor<i8>) -> Tensor<i8> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<i8>` using the `Div` trait.
impl i8TensorDiv of Div<Tensor<i8>> {
    /// Divides two `Tensor<i8>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<i8>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<i8>, rhs: Tensor<i8>) -> Tensor<i8> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

// Implements the Into trait for i8 tensor to i32 tensor.
impl TensorI8IntoTensorI32 of Into<Tensor<i8>, Tensor<i32>> {
    fn into(self: Tensor<i8>) -> Tensor<i32> {
        tensor_i8_to_tensor_i32(@self)
    }
}

// Implements the Into trait for i8 tensor to fp tensor.
impl TensorI8IntoTensorFP16x16 of Into<Tensor<i8>, Tensor<FP16x16>> {
    fn into(self: Tensor<i8>) -> Tensor<FP16x16> {
        tensor_i8_to_fp16x16(@self)
    }
}

fn tensor_i8_to_tensor_i32(x: @Tensor<i8>) -> Tensor<i32> {
    let mut result_data = ArrayTrait::<i32>::new();
    let mut data = *x.data;

    loop {
        result_data.append((*data.pop_front().unwrap()).into());

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span(), *x.extra);
}


fn tensor_i8_to_fp16x16(x: @Tensor<i8>) -> Tensor<FP16x16> {
    let mut result_data = ArrayTrait::<FP16x16>::new();
    let mut data = *x.data;

    loop {
        result_data.append(i8_to_fp16x16(*data.pop_front().unwrap()));

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span(), *x.extra);
}

/// Implements partial equal for two `Tensor<i8>` using the `PartialEq` trait.
impl i8TensorPartialEq of PartialEq<Tensor<i8>> {
    fn eq(lhs: @Tensor<i8>, rhs: @Tensor<i8>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<i8>, rhs: @Tensor<i8>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}
