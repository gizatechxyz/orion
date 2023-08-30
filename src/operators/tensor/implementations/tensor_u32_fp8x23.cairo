//! This module defines and implement a Tensor for u32 values.

use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedImpl, FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, ExtraParams, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor, tensor_eq
};
use orion::operators::tensor::math;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;
use orion::operators::tensor::implementations::tensor_i32_fp8x23::Tensor_i32_fp8x23;

impl Tensor_u32_fp8x23 of TensorTrait<u32, FP8x23> {
    fn new(shape: Span<usize>, data: Span<u32>, extra: Option<ExtraParams>) -> Tensor<u32> {
        new_tensor(shape, data, extra)
    }

    fn at(self: @Tensor<u32>, indices: Span<usize>) -> u32 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<u32>) -> u32 {
        math::min::min_in_tensor(*self.data)
    }

    fn max(self: @Tensor<u32>) -> u32 {
        math::max::max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<u32>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<u32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<u32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<u32>, target_shape: Span<usize>) -> Tensor<u32> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<u32>, axis: usize, keepdims: bool) -> Tensor<u32> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<u32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax::<u32, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<u32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin::<u32, FP8x23, u32>(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<u32>, axes: Span<usize>) -> Tensor<u32> {
        //transpose(self, axes)
        panic(array![])
    }

    fn matmul(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<u32> {
        //matmul(self, other)
        panic(array![])
    }

    fn exp(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::exp::exp_from_int::<u32, FP8x23, u32>(*self)
    }

    fn log(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::log::log_from_int::<u32, FP8x23, u32>(*self)
    }

    fn equal(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<u32>) -> Tensor<u32> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<u32>) -> Tensor<u32> {
        panic(array!['not supported with u32'])
    }

    fn sin(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::sin::sin_from_int(*self)
    }

    fn cos(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::cos::cos_from_int(*self)
    }

    fn asin(self: @Tensor<u32>) -> Tensor<FP8x23> {
        panic(array!['not supported with u32'])
    }

    fn cumsum(
        self: @Tensor<u32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<u32> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<u32>, axis: usize) -> Tensor<u32> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::sin::sin_from_int(*self)
    }

    fn tanh(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::tanh::tanh_from_int(*self)
    }

    fn cosh(self: @Tensor<u32>) -> Tensor<FP8x23> {
        math::cosh::cosh_from_int(*self)
    }

    fn acosh(self: @Tensor<u32>) -> Tensor<FP8x23> {
        // acosh_u32(self).unwrap()
        panic(array![])
    }

    fn asinh(self: @Tensor<u32>) -> Tensor<FP8x23> {
        // asinh_u32(self).unwrap()
        panic(array![])
    }

    fn atan(self: @Tensor<u32>) -> Tensor<FP8x23> {
        // atan_u32(self).unwrap()
        panic(array![])
    }

    fn xor(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        // xor(self, other)
        panic(array![])
    }

    fn or(self: @Tensor<u32>, other: @Tensor<u32>) -> Tensor<usize> {
        // or(self, other)
        panic(array![])
    }
    fn acos(self: @Tensor<u32>) -> Tensor<FP8x23> {
        panic(array!['not supported with u32'])
    }

    fn onehot(
        self: @Tensor<u32>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<u32> {
        // onehot(self, depth, axis, values)
        panic(array![])
    }

    fn sqrt(self: @Tensor<u32>) -> Tensor<FP8x23> {
        // sqrt_u32(self).unwrap()
        panic(array![])
    }

    fn concat(tensors: Span<Tensor<u32>>, axis: usize,) -> Tensor<u32> {
        // concat_u32(tensors, axis)
        panic(array![])
    }
}

/// Implements addition for `Tensor<u32>` using the `Add` trait.
impl u32TensorAdd of Add<Tensor<u32>> {
    /// Adds two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<u32>` using the `Sub` trait.
impl u32TensorSub of Sub<Tensor<u32>> {
    /// Subtracts two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<u32>` using the `Mul` trait.
impl u32TensorMul of Mul<Tensor<u32>> {
    /// Multiplies two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<u32>` using the `Div` trait.
impl u32TensorDiv of Div<Tensor<u32>> {
    /// Divides two `Tensor<u32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<u32>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<u32>, rhs: Tensor<u32>) -> Tensor<u32> {
        math::arithmetic::div(@lhs, @rhs)
    }
}


// Implements the Into trait for u32 tensor to fp tensor.
impl Tensoru32IntoTensorFP8x23 of Into<Tensor<u32>, Tensor<FP8x23>> {
    fn into(self: Tensor<u32>) -> Tensor<FP8x23> {
        tensor_u32_to_fp8x23(@self)
    }
}


fn tensor_u32_to_fp8x23(x: @Tensor<u32>) -> Tensor<FP8x23> {
    let mut result_data = ArrayTrait::<FP8x23>::new();
    let mut data = *x.data;

    loop {
        result_data.append(FixedTrait::<FP8x23>::new_unscaled(*data.pop_front().unwrap(), false));

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::<FP8x23, FP8x23>::new(*x.shape, result_data.span(), *x.extra);
}

/// Implements partial equal for two `Tensor<u32>` using the `PartialEq` trait.
impl u32TensorPartialEq of PartialEq<Tensor<u32>> {
    fn eq(lhs: @Tensor<u32>, rhs: @Tensor<u32>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<u32>, rhs: @Tensor<u32>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}
