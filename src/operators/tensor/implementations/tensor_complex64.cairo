use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{
    new_tensor, constant_of_shape, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core as core_tensor, ml, manipulation};
use orion::numbers::{NumberTrait, FP64x64, FP64x64Impl};
use orion::numbers::fixed_point::implementations::fp64x64::core::ONE;
use orion::operators::tensor::implementations::{
    tensor_i8::I8Tensor, tensor_u32::U32Tensor, tensor_bool::BoolTensor
};
use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::complex_number::complex64::{Complex64Impl, complex64};

impl Complex64Tensor of TensorTrait<complex64> {
    fn new(shape: Span<usize>, data: Span<complex64>) -> Tensor<complex64> {
        new_tensor(shape, data)
    }

    fn constant_of_shape(shape: Span<usize>, value: complex64) -> Tensor<complex64> {
        constant_of_shape(shape, value)
    }

    fn at(self: @Tensor<complex64>, indices: Span<usize>) -> complex64 {
        *at_tensor(self, indices)
    }

    fn add(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::add(@lhs, @rhs)
    }

    fn sub(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::sub(@lhs, @rhs)
    }

    fn mul(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::mul(@lhs, @rhs)
    }

    fn div(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::div(@lhs, @rhs)
    }

    fn min_in_tensor(self: @Tensor<complex64>) -> complex64 {
        panic(array!['not supported!'])
    }

    fn min(tensors: Span<Tensor<complex64>>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn max_in_tensor(self: @Tensor<complex64>) -> complex64 {
        panic(array!['not supported!'])
    }

    fn max(tensors: Span<Tensor<complex64>>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn stride(self: @Tensor<complex64>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<complex64>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<complex64>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(
        self: @Tensor<complex64>, target_shape: Span<i32>, allowzero: bool
    ) -> Tensor<complex64> {
        reshape(self, target_shape, allowzero)
    }

    fn reduce_sum(
        self: @Tensor<complex64>,
        axes: Option<Span<i32>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<complex64> {
        math::reduce_sum::reduce_sum(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_prod(self: @Tensor<complex64>, axis: usize, keepdims: bool) -> Tensor<complex64> {
        math::reduce_prod::reduce_prod(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<complex64>,
        axis: i32,
        keepdims: Option<bool>,
        select_last_index: Option<bool>
    ) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn argmin(
        self: @Tensor<complex64>,
        axis: usize,
        keepdims: Option<bool>,
        select_last_index: Option<bool>
    ) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn transpose(self: @Tensor<complex64>, axes: Span<usize>) -> Tensor<complex64> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<complex64> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::exp::exp(*self)
    }

    fn log(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::log::log(*self)
    }

    fn equal(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn greater_equal(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn less(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn less_equal(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn abs(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::abs::abs(*self)
    }

    fn neg(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn ceil(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn sin(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::sin::sin(*self)
    }

    fn cos(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::cos::cos(*self)
    }

    fn asin(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<complex64>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<complex64> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<complex64>, axis: usize) -> Tensor<complex64> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::sinh::sinh(*self)
    }

    fn tanh(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::tanh::tanh(*self)
    }

    fn cosh(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::cosh::cosh(*self)
    }

    fn acosh(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::acosh::acosh(*self)
    }

    fn asinh(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::asinh::asinh(*self)
    }

    fn atan(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::atan::atan(*self)
    }

    fn xor(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn or(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<usize> {
        panic(array!['not supported!'])
    }

    fn acos(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<complex64>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn sqrt(self: @Tensor<complex64>) -> Tensor<complex64> {
        math::sqrt::sqrt(*self)
    }

    fn concat(tensors: Span<Tensor<complex64>>, axis: usize,) -> Tensor<complex64> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<complex64>, y_scale: @Tensor<complex64>, y_zero_point: @Tensor<complex64>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<complex64>, x_zero_point: @Tensor<complex64>
    ) -> Tensor::<complex64> {
        panic(array!['not supported!'])
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<complex64>,
        a_zero_point: @Tensor<complex64>,
        b: @Tensor<i8>,
        b_scale: @Tensor<complex64>,
        b_zero_point: @Tensor<complex64>,
        y_scale: @Tensor<complex64>,
        y_zero_point: @Tensor<complex64>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_mul(
        self: @Tensor<i8>,
        a_scale: @Tensor<complex64>,
        a_zero_point: @Tensor<complex64>,
        b: @Tensor<i8>,
        b_scale: @Tensor<complex64>,
        b_zero_point: @Tensor<complex64>,
        y_scale: @Tensor<complex64>,
        y_zero_point: @Tensor<complex64>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_matmul(
        self: @Tensor<i8>,
        a_scale: @Tensor<complex64>,
        a_zero_point: @Tensor<complex64>,
        b: @Tensor<i8>,
        b_scale: @Tensor<complex64>,
        b_zero_point: @Tensor<complex64>,
        y_scale: @Tensor<complex64>,
        y_zero_point: @Tensor<complex64>
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_concat(
        tensors: Span<Tensor<i8>>,
        scales: Span<Tensor<complex64>>,
        zero_points: Span<Tensor<complex64>>,
        y_scale: @Tensor<complex64>,
        y_zero_point: @Tensor<complex64>,
        axis: usize
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn qlinear_leakyrelu(
        self: @Tensor<i8>,
        a_scale: @Tensor<complex64>,
        a_zero_point: @Tensor<complex64>,
        alpha: complex64
    ) -> Tensor::<i8> {
        panic(array!['not supported!'])
    }

    fn slice(
        self: @Tensor<complex64>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<complex64> {
        core_tensor::slice::<complex64>(self, starts, ends, axes, steps)
    }

    fn gather(
        self: @Tensor<complex64>, indices: Tensor<i32>, axis: Option<i32>
    ) -> Tensor<complex64> {
        math::gather::gather(self, indices, axis)
    }

    fn gather_nd(
        self: @Tensor<complex64>, indices: Tensor<usize>, batch_dims: Option<usize>
    ) -> Tensor<complex64> {
        math::gather_nd::gather_nd(self, indices, batch_dims)
    }

    fn nonzero(self: @Tensor<complex64>) -> Tensor<usize> {
        core_tensor::nonzero(self)
    }

    fn squeeze(self: @Tensor<complex64>, axes: Option<Span<usize>>) -> Tensor<complex64> {
        core_tensor::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<complex64>, axes: Span<usize>) -> Tensor<complex64> {
        core_tensor::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn clip(
        self: @Tensor<complex64>, min: Option<complex64>, max: Option<complex64>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<complex64>) -> Tensor<complex64> {
        core_tensor::identity(self)
    }

    fn where(
        self: @Tensor<complex64>, x: @Tensor<complex64>, y: @Tensor<complex64>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn bitwise_and(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn bitwise_or(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn round(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn reduce_l1(self: @Tensor<complex64>, axis: usize, keepdims: bool) -> Tensor<complex64> {
        math::reduce_l1::reduce_l1(self, axis, keepdims)
    }

    fn array_feature_extractor(
        self: @Tensor<complex64>, indices: Tensor<usize>
    ) -> Tensor<complex64> {
        ml::array_feature_extractor::array_feature_extractor(*self, indices)
    }

    fn binarizer(self: @Tensor<complex64>, threshold: Option<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn reduce_sum_square(
        self: @Tensor<complex64>, axis: usize, keepdims: bool
    ) -> Tensor<complex64> {
        math::reduce_sum_square::reduce_sum_square(self, axis, keepdims)
    }

    fn reduce_l2(self: @Tensor<complex64>, axis: usize, keepdims: bool) -> Tensor<complex64> {
        math::reduce_l2::reduce_l2_complex(self, axis, keepdims)
    }

    fn trilu(self: @Tensor<complex64>, upper: bool, k: i64) -> Tensor<complex64> {
        linalg::trilu::trilu(self, upper, k)
    }

    fn scatter(
        self: @Tensor<complex64>,
        updates: Tensor<complex64>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn not(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }


    fn gather_elements(
        self: @Tensor<complex64>, indices: Tensor<i32>, axis: Option<i32>
    ) -> Tensor<complex64> {
        math::gather_elements::gather_elements(self, indices, axis)
    }

    fn shrink(
        self: Tensor<complex64>, bias: Option<complex64>, lambd: Option<complex64>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn reduce_mean(
        self: @Tensor<complex64>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<complex64> {
        math::reduce_mean::reduce_mean(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_min(
        self: @Tensor<complex64>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn pow(self: @Tensor<complex64>, other: @Tensor<complex64>) -> Tensor<complex64> {
        math::pow::pow(self, other)
    }

    fn is_inf(
        self: @Tensor<complex64>, detect_negative: Option<u8>, detect_positive: Option<u8>
    ) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn is_nan(self: @Tensor<complex64>) -> Tensor<bool> {
        panic(array!['not supported!'])
    }

    fn reduce_log_sum(self: @Tensor<complex64>, axis: usize, keepdims: bool) -> Tensor<complex64> {
        math::reduce_log_sum::reduce_log_sum(self, axis, keepdims)
    }

    fn erf(self: @Tensor<complex64>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn unique(
        self: @Tensor<complex64>, axis: Option<usize>, sorted: Option<bool>
    ) -> (Tensor<complex64>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        panic(array!['not supported!'])
    }

    fn compress(
        self: @Tensor<complex64>, condition: Tensor<usize>, axis: Option<usize>
    ) -> Tensor<complex64> {
        math::compress::compress(self, condition, axis)
    }

    fn reduce_log_sum_exp(
        self: @Tensor<complex64>, axis: usize, keepdims: bool
    ) -> Tensor<complex64> {
        math::reduce_log_sum_exp::reduce_log_sum_exp(self, axis, keepdims)
    }

    fn instance_normalization(
        self: @Tensor<complex64>,
        scale: @Tensor<complex64>,
        bias: @Tensor<complex64>,
        epsilon: Option<complex64>,
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn layer_normalization(
        self: @Tensor<complex64>,
        scale: @Tensor<complex64>,
        B: Option<@Tensor<complex64>>,
        axis: Option<i32>,
        epsilon: Option<complex64>,
        stash_type: Option<usize>,
    ) -> (Tensor<complex64>, Tensor<complex64>, Tensor<complex64>) {
        panic(array!['not supported!'])
    }

    fn split(
        self: @Tensor<complex64>,
        axis: usize,
        num_outputs: Option<usize>,
        spl: Option<Tensor<usize>>
    ) -> Array<Tensor<complex64>> {
        manipulation::split::split(self, axis, num_outputs, spl)
    }

    fn reverse_sequence(
        self: @Tensor<complex64>,
        sequence_lens: Tensor<usize>,
        batch_axis: Option<usize>,
        time_axis: Option<usize>
    ) -> Tensor<complex64> {
        manipulation::reverse_sequence::reverse_sequence(self, sequence_lens, batch_axis, time_axis)
    }

    fn resize(
        self: @Tensor<complex64>,
        roi: Option<Tensor<complex64>>,
        scales: Option<Span<complex64>>,
        sizes: Option<Span<usize>>,
        antialias: Option<usize>,
        axes: Option<Span<usize>>,
        coordinate_transformation_mode: Option<math::resize::TRANSFORMATION_MODE>,
        cubic_coeff_a: Option<complex64>,
        exclude_outside: Option<bool>,
        extrapolation_value: Option<complex64>,
        keep_aspect_ratio_policy: Option<math::resize::KEEP_ASPECT_RATIO_POLICY>,
        mode: Option<math::resize::MODE>,
        nearest_mode: Option<math::resize::NEAREST_MODE>,
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn random_uniform_like(
        tensor: @Tensor<complex64>,
        high: Option<complex64>,
        low: Option<complex64>,
        seed: Option<usize>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn range(start: complex64, end: complex64, step: complex64) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn hann_window(size: complex64, periodic: Option<usize>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn hamming_window(size: complex64, periodic: Option<usize>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn blackman_window(size: complex64, periodic: Option<usize>) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn split_to_sequence(
        self: @Tensor<complex64>, axis: usize, keepdims: usize, split: Option<Tensor<usize>>
    ) -> Array<Tensor<complex64>> {
        manipulation::split_to_sequence::split_to_sequence(self, axis, keepdims, split)
    }

    fn optional(self: @Tensor<complex64>) -> Option<Tensor<complex64>> {
        manipulation::optional::optional(self)
    }

    fn dynamic_quantize_linear(
        self: @Tensor<complex64>
    ) -> (Tensor::<u32>, Tensor::<complex64>, Tensor<complex64>) {
        panic(array!['not supported!'])
    }

    fn scatter_nd(
        self: @Tensor<complex64>,
        updates: Tensor<complex64>,
        indices: Tensor<usize>,
        reduction: Option<usize>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }

    fn label_encoder(
        self: @Tensor<complex64>,
        default_list: Option<Span<complex64>>,
        default_tensor: Option<Tensor<complex64>>,
        keys: Option<Span<complex64>>,
        keys_tensor: Option<Tensor<complex64>>,
        values: Option<Span<complex64>>,
        values_tensor: Option<Tensor<complex64>>
    ) -> Tensor<complex64> {
        panic(array!['not supported!'])
    }
}

/// Implements addition for `Tensor<complex64>` using the `Add` trait.
impl Complex64TensorAdd of Add<Tensor<complex64>> {
    /// Adds two `Tensor<complex64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<complex64>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<complex64>` using the `Sub` trait.
impl Complex64TensorSub of Sub<Tensor<complex64>> {
    /// Subtracts two `Tensor<complex64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<complex64>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<complex64>` using the `Mul` trait.
impl Complex64TensorMul of Mul<Tensor<complex64>> {
    /// Multiplies two `Tensor<complex64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<complex64>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<complex64>` using the `Div` trait.
impl Complex64TensorDiv of Div<Tensor<complex64>> {
    /// Divides two `Tensor<complex64>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<complex64>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<complex64>, rhs: Tensor<complex64>) -> Tensor<complex64> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<complex64>` using the `complex64` trait.
impl Complex64TensorPartialEq of PartialEq<Tensor<complex64>> {
    fn eq(lhs: @Tensor<complex64>, rhs: @Tensor<complex64>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<complex64>, rhs: @Tensor<complex64>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

// Internals
fn eq(lhs: @complex64, rhs: @complex64) -> bool {
    let eq = (*lhs.real == *rhs.real) && (*lhs.img == *rhs.img);

    eq
}

fn tensor_eq(mut lhs: Tensor<complex64>, mut rhs: Tensor<complex64>,) -> bool {
    let mut is_eq = true;

    while lhs.shape.len() != 0
        && is_eq {
            is_eq = lhs.shape.pop_front().unwrap() == rhs.shape.pop_front().unwrap();
        };

    if !is_eq {
        return false;
    }

    while lhs.data.len() != 0
        && is_eq {
            is_eq = eq(lhs.data.pop_front().unwrap(), rhs.data.pop_front().unwrap());
        };

    is_eq
}

