use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, constant_of_shape, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, manipulation, core};
use orion::numbers::{i8, i32, NumberTrait, FP8x23W};
use orion::operators::tensor::implementations::{tensor_i8::I8Tensor, tensor_u32::U32Tensor};

impl FP8x23WTensor of TensorTrait<FP8x23W> {
    fn new(shape: Span<usize>, data: Span<FP8x23W>) -> Tensor<FP8x23W> {
        new_tensor(shape, data)
    }

    fn constant_of_shape(shape: Span<usize>, value: FP8x23W) -> Tensor<FP8x23W> {
        constant_of_shape(shape, value)
    }

    fn at(self: @Tensor<FP8x23W>, indices: Span<usize>) -> FP8x23W {
        *at_tensor(self, indices)
    }

    fn min_in_tensor(self: @Tensor<FP8x23W>) -> FP8x23W {
        math::min_in_tensor::min_in_tensor::<FP8x23W, u64>(*self.data)
    }

    fn min(tensors: Span<Tensor<FP8x23W>>) -> Tensor<FP8x23W> {
        math::min::min(tensors)
    }

    fn max_in_tensor(self: @Tensor<FP8x23W>) -> FP8x23W {
        math::max_in_tensor::max_in_tensor(*self.data)
    }

    fn max(tensors: Span<Tensor<FP8x23W>>) -> Tensor<FP8x23W> {
        math::max::max(tensors)
    }

    fn stride(self: @Tensor<FP8x23W>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FP8x23W>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FP8x23W>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<FP8x23W>, target_shape: Span<usize>) -> Tensor<FP8x23W> {
        reshape(self, target_shape)
    }

    fn reduce_sum(self: @Tensor<FP8x23W>, axis: usize, keepdims: bool) -> Tensor<FP8x23W> {
        math::reduce_sum::reduce_sum(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<FP8x23W>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<FP8x23W>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<FP8x23W>, axes: Span<usize>) -> Tensor<FP8x23W> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::exp::exp(*self)
    }

    fn log(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::log::log(*self)
    }

    fn equal(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::abs::abs(*self)
    }

    fn neg(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::neg::neg(*self)
    }

    fn ceil(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::ceil::ceil(*self)
    }

    fn sin(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::sin::sin(*self)
    }

    fn cos(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::cos::cos(*self)
    }

    fn asin(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<FP8x23W>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP8x23W> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<FP8x23W>, axis: usize) -> Tensor<FP8x23W> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::sinh::sinh(*self)
    }

    fn tanh(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::tanh::tanh(*self)
    }

    fn cosh(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::cosh::cosh(*self)
    }

    fn acosh(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::acosh::acosh(*self)
    }

    fn asinh(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::asinh::asinh(*self)
    }

    fn atan(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::atan::atan(*self)
    }

    fn xor(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<FP8x23W>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP8x23W> {
        panic(array!['not supported!'])
    }

    fn sqrt(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::sqrt::sqrt(*self)
    }

    fn concat(tensors: Span<Tensor<FP8x23W>>, axis: usize,) -> Tensor<FP8x23W> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<FP8x23W>, y_scale: @Tensor<FP8x23W>, y_zero_point: @Tensor<FP8x23W>
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
        self: @Tensor<i8>, x_scale: @Tensor<FP8x23W>, x_zero_point: @Tensor<FP8x23W>
    ) -> Tensor::<FP8x23W> {
        panic(array!['not supported!'])
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<FP8x23W>,
        a_zero_point: @Tensor<FP8x23W>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP8x23W>,
        b_zero_point: @Tensor<FP8x23W>,
        y_scale: @Tensor<FP8x23W>,
        y_zero_point: @Tensor<FP8x23W>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_matmul(
        self: @Tensor<i8>,
        a_scale: @Tensor<FP8x23W>,
        a_zero_point: @Tensor<FP8x23W>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP8x23W>,
        b_zero_point: @Tensor<FP8x23W>,
        y_scale: @Tensor<FP8x23W>,
        y_zero_point: @Tensor<FP8x23W>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn slice(
        self: @Tensor<FP8x23W>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<FP8x23W> {
        core::slice::<FP8x23W>(self, starts, ends, axes, steps)
    }

    fn gather(
        self: @Tensor<FP8x23W>, indices: Tensor<usize>, axis: Option<usize>
    ) -> Tensor<FP8x23W> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<FP8x23W>) -> Tensor<usize> {
        core::nonzero(self)
    }

    fn squeeze(self: @Tensor<FP8x23W>, axes: Option<Span<i32>>) -> Tensor<FP8x23W> {
        core::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<FP8x23W>, axes: Span<usize>) -> Tensor<FP8x23W> {
        core::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::sign::sign(*self)
    }

    fn clip(self: @Tensor<FP8x23W>, min: Option<FP8x23W>, max: Option<FP8x23W>) -> Tensor<FP8x23W> {
        core::clip(self, min, max)
    }

    fn and(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<usize> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        core::identity(self)
    }

    fn where(self: @Tensor<FP8x23W>, x: @Tensor<FP8x23W>, y: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::where::where(self, x, y)
    }

    fn bitwise_and(self: @Tensor<FP8x23W>, other: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::bitwise_and::bitwise_and(self, other)
    }

    fn round(self: @Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::round::round(*self)
    }

    fn reduce_l1(self: @Tensor<FP8x23W>, axis: usize, keepdims: bool) -> Tensor<FP8x23W> {
        math::reduce_l1::reduce_l1(self, axis, keepdims)
    }

    fn reduce_sum_square(self: @Tensor<FP8x23W>, axis: usize, keepdims: bool) -> Tensor<FP8x23W> {
        math::reduce_sum_square::reduce_sum_square(self, axis, keepdims)
    }

    fn reduce_l2(self: @Tensor<FP8x23W>, axis: usize, keepdims: bool) -> Tensor<FP8x23W> {
        math::reduce_l2::reduce_l2(self, axis, keepdims)
    }

    fn trilu(self: @Tensor<FP8x23W>, upper: bool, k: i64) -> Tensor<FP8x23W> {
        linalg::trilu::trilu(self, upper, k)
    }

    fn scatter(
        self: @Tensor<FP8x23W>,
        updates: Tensor<FP8x23W>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<FP8x23W> {
        math::scatter::scatter(self, updates, indices, axis, reduction)
    }

    fn unique(
        self: @Tensor<FP8x23W>, axis: Option<usize>, sorted: Option<bool>
    ) -> (Tensor<FP8x23W>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        manipulation::unique::unique(self, axis, sorted)
    }
}

/// Implements addition for `Tensor<FP8x23W>` using the `Add` trait.
impl FP8x23WTensorAdd<
    FP8x23W,
    impl FP8x23WTensor: TensorTrait<FP8x23W>,
    impl TAdd: Add<FP8x23W>,
    impl TCopy: Copy<FP8x23W>,
    impl TDrop: Drop<FP8x23W>
> of Add<Tensor<FP8x23W>> {
    /// Adds two `Tensor<FP8x23W>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23W>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FP8x23W>, rhs: Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FP8x23W>` using the `Sub` trait.
impl FP8x23WTensorSub<
    FP8x23W,
    impl FP8x23WTensor: TensorTrait<FP8x23W>,
    impl TSub: Sub<FP8x23W>,
    impl TCopy: Copy<FP8x23W>,
    impl TDrop: Drop<FP8x23W>
> of Sub<Tensor<FP8x23W>> {
    /// Subtracts two `Tensor<FP8x23W>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23W>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FP8x23W>, rhs: Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FP8x23W>` using the `Mul` trait.
impl FP8x23WTensorMul<
    FP8x23W,
    impl FP8x23WTensor: TensorTrait<FP8x23W>,
    impl TMul: Mul<FP8x23W>,
    impl TCopy: Copy<FP8x23W>,
    impl TDrop: Drop<FP8x23W>
> of Mul<Tensor<FP8x23W>> {
    /// Multiplies two `Tensor<FP8x23W>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23W>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FP8x23W>, rhs: Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FP8x23W>` using the `Div` trait.
impl FP8x23WTensorDiv<
    FP8x23W,
    impl FP8x23WTensor: TensorTrait<FP8x23W>,
    impl TDiv: Div<FP8x23W>,
    impl TCopy: Copy<FP8x23W>,
    impl TDrop: Drop<FP8x23W>
> of Div<Tensor<FP8x23W>> {
    /// Divides two `Tensor<FP8x23W>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP8x23W>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FP8x23W>, rhs: Tensor<FP8x23W>) -> Tensor<FP8x23W> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<FP8x23W>` using the `PartialEq` trait.
impl FP8x23WTensorPartialEq of PartialEq<Tensor<FP8x23W>> {
    fn eq(lhs: @Tensor<FP8x23W>, rhs: @Tensor<FP8x23W>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<FP8x23W>, rhs: @Tensor<FP8x23W>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl U32TryIntoU32 of TryInto<u64, u64> {
    fn try_into(self: u64) -> Option<u64> {
        Option::Some(self)
    }
}

// Internals

const PRECISION: u64 = 75497; // 0.009

fn relative_eq(lhs: @FP8x23W, rhs: @FP8x23W) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs.mag != 0 {
        (diff / *lhs).mag
    } else {
        diff.mag
    };

    rel_diff <= PRECISION
}

fn tensor_eq(mut lhs: Tensor<FP8x23W>, mut rhs: Tensor<FP8x23W>,) -> bool {
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

