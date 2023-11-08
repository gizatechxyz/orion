use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape, at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core};
use orion::numbers::{i32, i8, NumberTrait};
use orion::operators::tensor::implementations::{tensor_u32::U32Tensor, tensor_i8::I8Tensor};


impl I32Tensor of TensorTrait<i32> {
    fn new(shape: Span<usize>, data: Span<i32>) -> Tensor<i32> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<i32>, indices: Span<usize>) -> i32 {
        *at_tensor(self, indices)
    }

    fn min_in_tensor(self: @Tensor<i32>) -> i32 {
        math::min_in_tensor::min_in_tensor::<i32, u32>(*self.data)
    }

    fn min(tensors: Span<Tensor<i32>>) -> Tensor<i32> {
        math::min::min(tensors)
    }

    fn max_in_tensor(self: @Tensor<i32>) -> i32 {
        math::max_in_tensor::max_in_tensor(*self.data)
    }

    fn max(tensors: Span<Tensor<i32>>) -> Tensor<i32> {
        math::max::max(tensors)
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
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<i32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<i32> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn log(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
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

    fn neg(self: @Tensor<i32>) -> Tensor<i32> {
        math::neg::neg(*self)
    }

    fn ceil(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn sin(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn cos(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn asin(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn cumsum(
        self: @Tensor<i32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<i32> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn tanh(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn cosh(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn acosh(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn asinh(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn atan(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn xor(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn onehot(
        self: @Tensor<i32>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn sqrt(self: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
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
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
    ) -> Tensor::<i32> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<i32>,
        a_zero_point: @Tensor<i32>,
        b: @Tensor<i8>,
        b_scale: @Tensor<i32>,
        b_zero_point: @Tensor<i32>,
        y_scale: @Tensor<i32>,
        y_zero_point: @Tensor<i32>
    ) -> Tensor::<i8> {
        quantization::qlinear_add::qlinear_add(
            self,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn qlinear_matmul(
        self: @Tensor<i8>,
        a_scale: @Tensor<i32>,
        a_zero_point: @Tensor<i32>,
        b: @Tensor<i8>,
        b_scale: @Tensor<i32>,
        b_zero_point: @Tensor<i32>,
        y_scale: @Tensor<i32>,
        y_zero_point: @Tensor<i32>
    ) -> Tensor::<i8> {
        quantization::qlinear_matmul::qlinear_matmul(
            self,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }


    fn slice(
        self: @Tensor<i32>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<i32> {
        core::slice::<i32>(self, starts, ends, axes, steps)
    }

    fn gather(self: @Tensor<i32>, indices: Tensor<usize>, axis: Option<usize>) -> Tensor<i32> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<i32>) -> Tensor<usize> {
        core::nonzero(self)
    }

    fn squeeze(self: @Tensor<i32>, axes: Option<Span<i32>>) -> Tensor<i32> {
        core::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<i32>, axes: Span<usize>) -> Tensor<i32> {
        core::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<i32>) -> Tensor<i32> {
        math::sign::sign(*self)
    }

    fn clip(self: @Tensor<i32>, min: Option<i32>, max: Option<i32>) -> Tensor<i32> {
        core::clip(self, min, max)
    }

    fn and(self: @Tensor<i32>, other: @Tensor<i32>) -> Tensor<usize> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<i32>) -> Tensor<i32> {
        core::identity(self)
    }

    fn where(self: @Tensor<i32>, x: @Tensor<i32>, y: @Tensor<i32>) -> Tensor<i32> {
        math::where::where(self, x, y)
    }

    fn round(self: @Tensor<i32>) -> Tensor<i32> {
        math::round::round(*self)
    }

    fn trilu(self: @Tensor<i32>, upper: bool, k: i64) -> Tensor<i32> {
        linalg::trilu::trilu(self, upper, k)
    }
    fn scatter(
        self: @Tensor<i32>,
        updates: Tensor<i32>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<i32> {
        math::scatter::scatter(self, updates, indices, axis, reduction)
    }
}

/// Implements addition for `Tensor<i32>` using the `Add` trait.
impl I32TensorAdd of Add<Tensor<i32>> {
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
impl I32TensorSub of Sub<Tensor<i32>> {
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
impl I32TensorMul of Mul<Tensor<i32>> {
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
impl I32TensorDiv of Div<Tensor<i32>> {
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
impl I32TensorPartialEq of PartialEq<Tensor<i32>> {
    fn eq(lhs: @Tensor<i32>, rhs: @Tensor<i32>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<i32>, rhs: @Tensor<i32>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl I8TryIntoI8 of TryInto<i32, i32> {
    fn try_into(self: i32) -> Option<i32> {
        Option::Some(self)
    }
}

impl TensorI8IntoTensorI32 of Into<Tensor<i8>, Tensor<i32>> {
    fn into(self: Tensor<i8>) -> Tensor<i32> {
        tensor_i8_to_tensor_i32(@self)
    }
}


// Internals

fn tensor_eq(mut lhs: Tensor<i32>, mut rhs: Tensor<i32>,) -> bool {
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

fn tensor_i8_to_tensor_i32(x: @Tensor<i8>) -> Tensor<i32> {
    let mut result_data = ArrayTrait::<i32>::new();
    let mut data = *x.data;

    loop {
        result_data.append((*data.pop_front().unwrap()).into());

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*x.shape, result_data.span());
}
