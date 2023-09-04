use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization};
use orion::numbers::{i8, NumberTrait, FP64x64, FP64x64Impl};
use orion::numbers::fixed_point::implementations::fp64x64::core::ONE;
use orion::operators::tensor::implementations::{tensor_i8::I8Tensor, tensor_u32::U32Tensor};

impl FP64x64Tensor of TensorTrait<FP64x64> {
    fn new(shape: Span<usize>, data: Span<FP64x64>) -> Tensor<FP64x64> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<FP64x64>, indices: Span<usize>) -> FP64x64 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<FP64x64>) -> FP64x64 {
        math::min::min_in_tensor::<FP64x64, u128>(*self.data)
    }

    fn max(self: @Tensor<FP64x64>) -> FP64x64 {
        math::max::max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<FP64x64>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FP64x64>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FP64x64>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<FP64x64>, target_shape: Span<usize>) -> Tensor<FP64x64> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<FP64x64>, axis: usize, keepdims: bool) -> Tensor<FP64x64> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<FP64x64>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<FP64x64>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<FP64x64>, axes: Span<usize>) -> Tensor<FP64x64> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::exp::exp(*self)
    }

    fn log(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::log::log(*self)
    }

    fn equal(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::ceil::ceil(*self)
    }

    fn sin(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::sin::sin(*self)
    }

    fn cos(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::cos::cos(*self)
    }

    fn asin(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<FP64x64>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP64x64> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<FP64x64>, axis: usize) -> Tensor<FP64x64> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::sinh::sinh(*self)
    }

    fn tanh(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::tanh::tanh(*self)
    }

    fn cosh(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::cosh::cosh(*self)
    }

    fn acosh(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::acosh::acosh(*self)
    }

    fn asinh(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::asinh::asinh(*self)
    }

    fn atan(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::atan::atan(*self)
    }

    fn xor(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<FP64x64>, other: @Tensor<FP64x64>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<FP64x64>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP64x64> {
        math::onehot::onehot(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::sqrt::sqrt(*self)
    }

    fn concat(tensors: Span<Tensor<FP64x64>>, axis: usize,) -> Tensor<FP64x64> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<FP64x64>, y_scale: @Tensor<FP64x64>, y_zero_point: @Tensor<FP64x64>
    ) -> Tensor::<i8> {
        quantization::quantize_linear::quantize_linear(
            self,
            y_scale,
            y_zero_point,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<FP64x64>, x_zero_point: @Tensor<FP64x64>
    ) -> Tensor::<FP64x64> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }
}

/// Implements addition for `Tensor<FP64x64>` using the `Add` trait.
impl FP64x64TensorAdd of Add<Tensor<FP64x64>> {
    /// Adds two `Tensor<FP64x64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP64x64>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FP64x64>, rhs: Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FP64x64>` using the `Sub` trait.
impl FP64x64TensorSub of Sub<Tensor<FP64x64>> {
    /// Subtracts two `Tensor<FP64x64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP64x64>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FP64x64>, rhs: Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FP64x64>` using the `Mul` trait.
impl FP64x64TensorMul of Mul<Tensor<FP64x64>> {
    /// Multiplies two `Tensor<FP64x64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP64x64>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FP64x64>, rhs: Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FP64x64>` using the `Div` trait.
impl FP64x64TensorDiv of Div<Tensor<FP64x64>> {
    /// Divides two `Tensor<FP64x64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP64x64>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FP64x64>, rhs: Tensor<FP64x64>) -> Tensor<FP64x64> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<FP64x64>` using the `PartialEq` trait.
impl FP64x64TensorPartialEq of PartialEq<Tensor<FP64x64>> {
    fn eq(lhs: @Tensor<FP64x64>, rhs: @Tensor<FP64x64>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<FP64x64>, rhs: @Tensor<FP64x64>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl U32TryIntoU128 of TryInto<u32, u128> {
    fn try_into(self: u32) -> Option<u128> {
        Option::Some(self.into())
    }
}

impl FP64x64TryIntoI8 of TryInto<FP64x64, i8> {
    fn try_into(self: FP64x64) -> Option<i8> {
        Option::Some(i8 { mag: (self.mag / ONE).try_into().unwrap(), sign: self.sign })
    }
}

impl TensorI8IntoTensorFP64x64 of Into<Tensor<i8>, Tensor<FP64x64>> {
    fn into(self: Tensor<i8>) -> Tensor<FP64x64> {
        tensor_i8_to_tensor_fp64x64(@self)
    }
}


// Internals

const PRECISION: u128 = 75497; // 0.009

fn relative_eq(lhs: @FP64x64, rhs: @FP64x64) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs.mag != 0 {
        (diff / *lhs).mag
    } else {
        diff.mag
    };

    rel_diff <= PRECISION
}

fn tensor_eq(mut lhs: Tensor<FP64x64>, mut rhs: Tensor<FP64x64>,) -> bool {
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

fn tensor_i8_to_tensor_fp64x64(x: @Tensor<i8>) -> Tensor<FP64x64> {
    let mut result_data = ArrayTrait::<FP64x64>::new();
    let mut data = *x.data;

    loop {
        result_data.append((*data.pop_front().unwrap()).into());

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span());
}
