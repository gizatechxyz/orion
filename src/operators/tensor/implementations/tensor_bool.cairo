use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;
use core::traits::{TryInto, Into};

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    constant_of_shape, new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core as core_ops, ml};
use orion::numbers::{i8, i32, NumberTrait};
use orion::operators::tensor::implementations::tensor_u32::U32Tensor;

impl BoolTensor of TensorTrait<bool> {
    fn new(shape: Span<usize>, data: Span<bool>) -> Tensor<bool> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<bool>, indices: Span<usize>) -> bool {
        *at_tensor(self, indices)
    }

    fn add(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn sub(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn mul(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn div(lhs: Tensor<bool>, rhs: Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }


    fn min_in_tensor(self: @Tensor<bool>) -> bool {
        panic(array!['not supported!'])
    }

    fn min(tensors: Span<Tensor<bool>>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn max_in_tensor(self: @Tensor<bool>) -> bool {
        panic(array!['not supported!'])
    }

    fn max(tensors: Span<Tensor<bool>>) -> Tensor<bool> {
        panic(array!['not supported!'])
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
        panic(array!['not supported!'])
    }

    fn reduce_prod(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn argmax(
        self: @Tensor<bool>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn argmin(
        self: @Tensor<bool>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn transpose(self: @Tensor<bool>, axes: Span<usize>) -> Tensor<bool> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
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
        panic(array!['not supported!'])
    }

    fn greater_equal(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn less(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn less_equal(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn abs(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn neg(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
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
        panic(array!['not supported!'])
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
        panic(array!['not supported!'])
    }

    fn or(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<usize> {
        panic(array!['not supported!'])
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
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<bool>, x_zero_point: @Tensor<bool>
    ) -> Tensor::<bool> {
        panic(array!['not supported!'])
    }

    fn slice(
        self: @Tensor<bool>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<bool> {
        core_ops::slice::<bool>(self, starts, ends, axes, steps)
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

    fn and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<bool>) -> Tensor<bool> {
        core_ops::identity(self)
    }

    fn where(self: @Tensor<bool>, x: @Tensor<bool>, y: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn qlinear_matmul(
        self: @Tensor<i8>,
        a_scale: @Tensor<bool>,
        a_zero_point: @Tensor<bool>,
        b: @Tensor<i8>,
        b_scale: @Tensor<bool>,
        b_zero_point: @Tensor<bool>,
        y_scale: @Tensor<bool>,
        y_zero_point: @Tensor<bool>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn not(self: @Tensor<bool>) -> Tensor<bool> {
        math::not::not(*self)
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<bool>,
        a_zero_point: @Tensor<bool>,
        b: @Tensor<i8>,
        b_scale: @Tensor<bool>,
        b_zero_point: @Tensor<bool>,
        y_scale: @Tensor<bool>,
        y_zero_point: @Tensor<bool>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_mul(
        self: @Tensor<i8>,
        a_scale: @Tensor<bool>,
        a_zero_point: @Tensor<bool>,
        b: @Tensor<i8>,
        b_scale: @Tensor<bool>,
        b_zero_point: @Tensor<bool>,
        y_scale: @Tensor<bool>,
        y_zero_point: @Tensor<bool>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_concat(
        tensors: Span<Tensor<i8>>,
        scales: Span<Tensor<bool>>,
        zero_points: Span<Tensor<bool>>,
        y_scale: @Tensor<bool>,
        y_zero_point: @Tensor<bool>,
        axis: usize
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }
    fn qlinear_leakyrelu(
        self: @Tensor<i8>, a_scale: @Tensor<bool>, a_zero_point: @Tensor<bool>, alpha: bool,
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }


    fn round(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn scatter(
        self: @Tensor<bool>,
        updates: Tensor<bool>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn trilu(self: @Tensor<bool>, upper: bool, k: i64) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn bitwise_and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn bitwise_or(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_l1(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_l2(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_sum_square(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn constant_of_shape(shape: Span<usize>, value: bool) -> Tensor<bool> {
        constant_of_shape(shape, value)
    }

    fn gather_elements(
        self: @Tensor<bool>, indices: Tensor<usize>, axis: Option<usize>
    ) -> Tensor<bool> {
        math::gather_elements::gather_elements(self, indices, axis)
    }

    fn sequence_length(self: Array<Tensor<bool>>) -> Tensor<u32> {
        math::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<bool>>, position: Tensor<i32>) -> Tensor<bool> {
        math::sequence_at::sequence_at(sequence, position)
    }

    fn shrink(self: Tensor<bool>, bias: Option<bool>, lambd: Option<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_mean(
        self: @Tensor<bool>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn binarizer(self: @Tensor<bool>, threshold: Option<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn array_feature_extractor(self: @Tensor<bool>, indices: Tensor<usize>) -> Tensor<bool> {
        ml::array_feature_extractor::array_feature_extractor(*self, indices)
    }

    fn reduce_min(
        self: @Tensor<bool>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn pow(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn sequence_erase(
        sequence: Array<Tensor<bool>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<bool>> {
        math::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<bool>>, tensor: @Tensor<bool>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<bool>> {
        math::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn is_inf(
        self: @Tensor<bool>, detect_negative: Option<u8>, detect_positive: Option<u8>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn is_nan(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<bool>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn erf(self: @Tensor<bool>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_log_sum(self: @Tensor<bool>, axis: usize, keepdims: bool) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn unique(
        self: @Tensor<bool>, axis: Option<usize>, sorted: Option<bool>
    ) -> (Tensor<bool>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        panic(array!['not supported!'])
    }

    fn gather_nd(self: @Tensor<bool>, indices: Tensor<usize>, batch_dims: Option<usize>) -> Tensor<bool> {
        math::gather_nd::gather_nd(self, indices, batch_dims)
    }
}

/// Implements partial equal for two `Tensor<bool>` using the `PartialEq` trait.
impl BoolTensorPartialEq of PartialEq<Tensor<bool>> {
    fn eq(lhs: @Tensor<bool>, rhs: @Tensor<bool>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<bool>, rhs: @Tensor<bool>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl BoolTryIntobool of TryInto<bool, bool> {
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
