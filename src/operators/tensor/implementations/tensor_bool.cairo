use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core};
use orion::numbers::{bool, i32, NumberTrait};
use orion::operators::tensor::implementations::tensor_u32::U32Tensor;

impl BoolTensor of TensorTrait<bool> {
    fn new(shape: Span<usize>, data: Span<bool>) -> Tensor<bool> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<bool>, indices: Span<usize>) -> bool {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<bool>) -> bool {
        math::min::min_in_tensor::<bool, u8>(*self.data)
    }

    fn max(self: @Tensor<bool>) -> bool {
        math::max::max_in_tensor(*self.data)
    }

    fn stride(self: @Tensor<bool>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<bool>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<bool>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<bool>, target_shape: Span<usize>) -> Tensor<bool> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<bool>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<bool>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<bool>, axes: Span<usize>) -> Tensor<bool> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn log(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn equal(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<bool>) -> Tensor<bool> {
        math::abs::abs(*self)
    }

    fn ceil(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn sin(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn cos(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn asin(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn cumsum(
        self: @Tensor<bool>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<bool> {
         panic(array!['not supported!'])
    }

    fn flatten(self: @Tensor<bool>, axis: usize) -> Tensor<bool> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn tanh(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn cosh(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn acosh(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn asinh(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn atan(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn xor(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn onehot(
        self: @Tensor<bool>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn sqrt(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn concat(tensors: Span<Tensor<bool>>, axis: usize,) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn quantize_linear(
        self: @Tensor<bool>, y_scale: @Tensor<bool>, y_zero_point: @Tensor<bool>
    ) -> Tensor::<bool> {
        quantization::quantize_linear::quantize_linear(
            self,
            y_scale,
            y_zero_point,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn dequantize_linear(
        self: @Tensor<bool>, x_scale: @Tensor<bool>, x_zero_point: @Tensor<bool>
    ) -> Tensor::<bool> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }

    fn slice(
        self: @Tensor<bool>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<bool> {
        core::slice::<bool>(self, starts, ends, axes, steps)
    }

    fn gather(self: @Tensor<bool>, indices: Tensor<usize>, axis: Option<usize>) -> Tensor<bool> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<bool>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn squeeze(self: @Tensor<bool>, axes: Option<Span<i32>>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn unsqueeze(self: @Tensor<bool>, axes: Span<usize>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn sign(self: @Tensor<bool>) -> Tensor<bool> {
         panic(array!['not supported!'])
    }

    fn clip(self: @Tensor<bool>, min: Option<bool>, max: Option<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }
}

/// Implements addition for `Tensor<bool>` using the `Add` trait.
impl boolTensorAdd of Add<Tensor<bool>> {
    /// Adds two `Tensor<bool>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<bool>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
         panic(array!['not supported!'])
    }
}

/// Implements subtraction for `Tensor<bool>` using the `Sub` trait.
impl boolTensorSub of Sub<Tensor<bool>> {
    /// Subtracts two `Tensor<bool>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<bool>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }
}

/// Implements multiplication for `Tensor<bool>` using the `Mul` trait.
impl boolTensorMul of Mul<Tensor<bool>> {
    /// Multiplies two `Tensor<bool>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<bool>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }
}

/// Implements division for `Tensor<bool>` using the `Div` trait.
impl boolTensorDiv of Div<Tensor<bool>> {
    /// Divides two `Tensor<bool>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<bool>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
         panic(array!['not supported!'])
    }
}

/// Implements partial equal for two `Tensor<bool>` using the `PartialEq` trait.
impl boolTensorPartialEq of PartialEq<Tensor<bool>> {
    fn eq(lhs: @Tensor<bool>, rhs: @Tensor<bool>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<bool>, rhs: @Tensor<bool>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl boolTryIntobool of TryInto<bool, bool> {
    fn try_into(self: bool) -> Option<bool> {
        Option::Some(self)
    }
}

// Internals

fn tensor_eq(mut lhs: Tensor<bool>, mut rhs: Tensor<bool>,) -> bool {
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

        is_eq = lhs.data.pop_front().unwrap() == rhs.data.pop_front().unwrap();
    };

    return is_eq;
}
