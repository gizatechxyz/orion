use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core};
use orion::numbers::{i8, i32, NumberTrait};
use orion::operators::tensor::implementations::tensor_u32::U32Tensor;

impl I8Tensor of TensorTrait<i8> {
    fn new(shape: Span<usize>, data: Span<i8>) -> Tensor<i8> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<i8>, indices: Span<usize>) -> i8 {
        *at_tensor(self, indices)
    }

    fn min(self: @Tensor<i8>) -> i8 {
        math::min::min_in_tensor::<i8, u8>(*self.data)
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
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i8>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i8>, axes: Span<usize>) -> Tensor<i8> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<i8> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn log(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
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
        panic(array!['not supported!'])
    }

    fn sin(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn cos(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn asin(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn cumsum(
        self: @Tensor<i8>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i8> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn tanh(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn cosh(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn acosh(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn asinh(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn atan(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn xor(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<i8>, other: @Tensor<i8>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn onehot(
        self: @Tensor<i8>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn sqrt(self: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn concat(tensors: Span<Tensor<i8>>, axis: usize,) -> Tensor<i8> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<i8>, y_scale: @Tensor<i8>, y_zero_point: @Tensor<i8>
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
        self: @Tensor<i8>, x_scale: @Tensor<i8>, x_zero_point: @Tensor<i8>
    ) -> Tensor::<i8> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }

    fn slice(
        self: @Tensor<i8>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<i8> {
        core::slice::<i8>(self, starts, ends, axes, steps)
    }

    fn gather(self: @Tensor<i8>, indices: Tensor<usize>, axis: Option<usize>) -> Tensor<i8> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<i8>) -> Tensor<usize> {
        core::nonzero(self)
    }

    fn squeeze(self: @Tensor<i8>, axes: Option<Span<i32>>) -> Tensor<i8> {
        core::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<i8>, axes: Span<usize>) -> Tensor<i8> {
        core::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<i8>) -> Tensor<i8> {
        math::sign::sign(*self)
    }

    fn clip(self: @Tensor<i8>, min: Option<i8>, max: Option<i8>) -> Tensor<i8> {
        core::clip(self, min, max)
    }

    fn identity(self: @Tensor<i8>) -> Tensor<i8> {
        core::identity(self)
    }

}

/// Implements addition for `Tensor<i8>` using the `Add` trait.
impl I8TensorAdd of Add<Tensor<i8>> {
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
impl I8TensorSub of Sub<Tensor<i8>> {
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
impl I8TensorMul of Mul<Tensor<i8>> {
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
impl I8TensorDiv of Div<Tensor<i8>> {
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

/// Implements partial equal for two `Tensor<i8>` using the `PartialEq` trait.
impl I8TensorPartialEq of PartialEq<Tensor<i8>> {
    fn eq(lhs: @Tensor<i8>, rhs: @Tensor<i8>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<i8>, rhs: @Tensor<i8>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl I8TryIntoI8 of TryInto<i8, i8> {
    fn try_into(self: i8) -> Option<i8> {
        Option::Some(self)
    }
}

// Internals

fn tensor_eq(mut lhs: Tensor<i8>, mut rhs: Tensor<i8>,) -> bool {
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
