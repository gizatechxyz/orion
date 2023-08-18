//! This module defines and implement a Tensor for i8 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::signed_integer::{i8::i8, i32::i32};
use orion::numbers::signed_integer::i8::{i8_to_fp8x23, i8_to_fp16x16};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor, tensor_eq
};
use orion::operators::tensor::math::min::min_i8::min_in_tensor;
use orion::operators::tensor::math::max::max_i8::max_in_tensor;
use orion::operators::tensor::math::reduce_sum::reduce_sum_i8::reduce_sum;
use orion::operators::tensor::math::argmax::argmax_i8::argmax;
use orion::operators::tensor::math::argmin::argmin_i8::argmin;
use orion::operators::tensor::math::equal::equal_i8::equal;
use orion::operators::tensor::math::greater::greater_i8::greater;
use orion::operators::tensor::math::greater_equal::greater_equal_i8::greater_equal;
use orion::operators::tensor::math::less::less_i8::less;
use orion::operators::tensor::math::less_equal::less_equal_i8::less_equal;
use orion::operators::tensor::math::abs::abs_i8::abs;
use orion::operators::tensor::linalg::matmul::matmul_i8::matmul;
use orion::operators::tensor::linalg::transpose::transpose_i8::transpose;
use orion::operators::tensor::math::exp::exp_i8::core::exp_i8;
use orion::operators::tensor::math::log::log_i8::core::log_i8;
use orion::operators::tensor::math::arithmetic::arithmetic_i8::{add, sub, mul, div};
use orion::operators::tensor::math::cumsum::cumsum_i8::cumsum;
use orion::operators::tensor::math::flatten::flatten_i8::flatten;
use orion::operators::tensor::math::sinh::sinh_i8::core::sinh_i8;
use orion::operators::tensor::math::tanh::tanh_i8::core::tanh_i8;
use orion::operators::tensor::math::cosh::cosh_i8::core::cosh_i8;
use orion::operators::tensor::math::acosh::acosh_i8::core::acosh_i8;
use orion::operators::tensor::math::asinh::asinh_i8::core::asinh_i8;
use orion::operators::tensor::math::sin::sin_i8::core::sin_i8;
use orion::operators::tensor::math::cos::cos_i8::core::cos_i8;
use orion::operators::tensor::math::atan::atan_i8::core::atan_i8;
use orion::operators::tensor::math::xor::xor_i8::xor;
use orion::operators::tensor::math::or::or_i8::or;
use orion::operators::tensor::math::onehot::onehot_i8::onehot;
use orion::operators::tensor::math::sqrt::sqrt_i8::core::sqrt_i8;
use orion::operators::tensor::math::concat::concat_i8::concat_i8;


impl Tensor_i8 of TensorTrait<i8> {
    fn new(shape: Span<usize>, data: Span<i8>, extra: Option<ExtraParams>) -> Tensor<i8> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<i8>, indices: Span<usize>) -> i8 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<i8>) -> i8 {
        min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<i8>) -> i8 {
        max_in_tensor(*self.data)
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
        reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<i8>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i8>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i8>, axes: Span<usize>) -> Tensor<i8> {
        transpose(self, axes)
    }

    fn matmul(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<i8> {
        matmul(self, other)
    }

    fn exp(self: @Tensor<i8>) -> Tensor<FixedType> {
        exp_i8(self).unwrap()
    }

    fn log(self: @Tensor<i8>) -> Tensor<FixedType> {
        log_i8(self).unwrap()
    }

    fn equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        equal(self, other)
    }

    fn greater(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        greater(self, other)
    }

    fn greater_equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        greater_equal(self, other)
    }

    fn less(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        less(self, other)
    }

    fn less_equal(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        less_equal(self, other)
    }

    fn abs(self: @Tensor<i8>) -> Tensor<i8> {
        abs(self)
    }

    fn ceil(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported with i8'])
    }

    fn sin(self: @Tensor<i8>) -> Tensor<FixedType> {
        sin_i8(self).unwrap()
    }

    fn cos(self: @Tensor<i8>) -> Tensor<FixedType> {
        cos_i8(self).unwrap()
    }

    fn asin(self: @Tensor<i8>) -> Tensor<FixedType> {
        panic(array!['not supported with i8'])
    }

    fn cumsum(
        self: @Tensor<i8>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i8> {
        cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        flatten(self, axis)
    }

    fn sinh(self: @Tensor<i8>) -> Tensor<FixedType> {
        sinh_i8(self).unwrap()
    }

    fn tanh(self: @Tensor<i8>) -> Tensor<FixedType> {
        tanh_i8(self).unwrap()
    }

    fn cosh(self: @Tensor<i8>) -> Tensor<FixedType> {
        cosh_i8(self).unwrap()
    }

    fn acosh(self: @Tensor<i8>) -> Tensor<FixedType> {
        acosh_i8(self).unwrap()
    }

    fn asinh(self: @Tensor<i8>) -> Tensor<FixedType> {
        asinh_i8(self).unwrap()
    }

    fn atan(self: @Tensor<i8>) -> Tensor<FixedType> {
        atan_i8(self).unwrap()
    }

    fn xor(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        xor(self, other)
    }

    fn or(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        or(self, other)
    }
    fn acos(self: @Tensor<i8>) -> Tensor<FixedType> {
        panic(array!['not supported with i8'])
    }

    fn onehot(
        self: @Tensor<i8>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i8> {
        onehot(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<i8>) -> Tensor<FixedType> {
        sqrt_i8(self).unwrap()
    }  

    fn concat( tensors: Span<Tensor<i8>>, axis: usize,  ) -> Tensor<i8> {
         concat_i8(tensors, axis)
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
        add(@lhs, @rhs)
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
        sub(@lhs, @rhs)
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
        mul(@lhs, @rhs)
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
        div(@lhs, @rhs)
    }
}

// Implements the Into trait for i8 tensor to i32 tensor.
impl TensorI8IntoTensorI32 of Into<Tensor<i8>, Tensor<i32>> {
    fn into(self: Tensor<i8>) -> Tensor<i32> {
        tensor_i8_to_tensor_i32(@self)
    }
}

// Implements the Into trait for i8 tensor to fp tensor.
impl TensorI8IntoTensorFP of Into<Tensor<i8>, Tensor<FixedType>> {
    fn into(self: Tensor<i8>) -> Tensor<FixedType> {
        tensor_i8_to_tensor_fp(@self).unwrap()
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

fn tensor_i8_to_tensor_fp(x: @Tensor<i8>) -> Option<Tensor<FixedType>> {
    match *x.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(tensor_i8_to_fp8x23(x)),
                FixedImpl::FP16x16(()) => Option::Some(tensor_i8_to_fp16x16(x)),
            },
            Option::None(_) => Option::Some(tensor_i8_to_fp16x16(x)),
        },
        Option::None(_) => Option::Some(tensor_i8_to_fp16x16(x)),
    }
}

fn tensor_i8_to_fp8x23(x: @Tensor<i8>) -> Tensor<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();
    let mut data = *x.data;

    loop {
        result_data.append(i8_to_fp8x23(*data.pop_front().unwrap()));

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span(), *x.extra);
}

fn tensor_i8_to_fp16x16(x: @Tensor<i8>) -> Tensor<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();
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
