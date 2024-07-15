use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::helpers::SpanPartialOrd;
use orion::operators::tensor::core::{
    new_tensor, constant_of_shape, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core as core_tensor, ml, manipulation};
use orion::numbers::{NumberTrait, FP32x32, FP32x32Impl};
use orion::numbers::fixed_point::implementations::fp32x32::core::ONE;
use orion::operators::tensor::implementations::{
    tensor_i8::I8Tensor, tensor_u32::U32Tensor, tensor_bool::BoolTensor
};

impl FP32x32Tensor of TensorTrait<FP32x32> {
    fn new(shape: Span<usize>, data: Span<FP32x32>) -> Tensor<FP32x32> {
        new_tensor(shape, data)
    }

    fn constant_of_shape(shape: Span<usize>, value: FP32x32) -> Tensor<FP32x32> {
        constant_of_shape(shape, value)
    }

    fn add(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::add(@lhs, @rhs)
    }

    fn sub(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::sub(@lhs, @rhs)
    }

    fn mul(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::mul(@lhs, @rhs)
    }

    fn div(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::div(@lhs, @rhs)
    }

    fn at(self: @Tensor<FP32x32>, indices: Span<usize>) -> FP32x32 {
        *at_tensor(self, indices)
    }

    fn min_in_tensor(self: @Tensor<FP32x32>) -> FP32x32 {
        math::min_in_tensor::min_in_tensor::<FP32x32, u64>(*self.data)
    }

    fn min(tensors: Span<Tensor<FP32x32>>) -> Tensor<FP32x32> {
        math::min::min(tensors)
    }

    fn max_in_tensor(self: @Tensor<FP32x32>) -> FP32x32 {
        math::max_in_tensor::max_in_tensor(*self.data)
    }

    fn max(tensors: Span<Tensor<FP32x32>>) -> Tensor<FP32x32> {
        math::max::max(tensors)
    }

    fn stride(self: @Tensor<FP32x32>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FP32x32>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FP32x32>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(
        self: @Tensor<FP32x32>, target_shape: Span<i32>, allowzero: bool
    ) -> Tensor<FP32x32> {
        reshape(self, target_shape, allowzero)
    }

    fn reduce_sum(
        self: @Tensor<FP32x32>,
        axes: Option<Span<i32>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP32x32> {
        math::reduce_sum::reduce_sum(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_prod(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_prod::reduce_prod(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<FP32x32>, axis: i32, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<i32> {
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<FP32x32>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<FP32x32>, axes: Span<usize>) -> Tensor<FP32x32> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::exp::exp(*self)
    }

    fn log(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::log::log(*self)
    }

    fn equal(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<i32> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<i32> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::abs::abs(*self)
    }

    fn neg(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::neg::neg(*self)
    }

    fn ceil(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::ceil::ceil(*self)
    }

    fn sin(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::sin::sin(*self)
    }

    fn cos(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::cos::cos(*self)
    }

    fn asin(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<FP32x32>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP32x32> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<FP32x32>, axis: usize) -> Tensor<FP32x32> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::sinh::sinh(*self)
    }

    fn tanh(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::tanh::tanh(*self)
    }

    fn cosh(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::cosh::cosh(*self)
    }

    fn acosh(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::acosh::acosh(*self)
    }

    fn asinh(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::asinh::asinh(*self)
    }

    fn atan(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::atan::atan(*self)
    }

    fn xor(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<FP32x32>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP32x32> {
        math::onehot::onehot(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::sqrt::sqrt(*self)
    }

    fn concat(tensors: Span<Tensor<FP32x32>>, axis: usize,) -> Tensor<FP32x32> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<FP32x32>, y_scale: @Tensor<FP32x32>, y_zero_point: @Tensor<FP32x32>
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
        self: @Tensor<i8>, x_scale: @Tensor<FP32x32>, x_zero_point: @Tensor<FP32x32>
    ) -> Tensor::<FP32x32> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<FP32x32>,
        a_zero_point: @Tensor<FP32x32>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP32x32>,
        b_zero_point: @Tensor<FP32x32>,
        y_scale: @Tensor<FP32x32>,
        y_zero_point: @Tensor<FP32x32>
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

    fn qlinear_mul(
        self: @Tensor<i8>,
        a_scale: @Tensor<FP32x32>,
        a_zero_point: @Tensor<FP32x32>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP32x32>,
        b_zero_point: @Tensor<FP32x32>,
        y_scale: @Tensor<FP32x32>,
        y_zero_point: @Tensor<FP32x32>
    ) -> Tensor::<i8> {
        quantization::qlinear_mul::qlinear_mul(
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
        a_scale: @Tensor<FP32x32>,
        a_zero_point: @Tensor<FP32x32>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP32x32>,
        b_zero_point: @Tensor<FP32x32>,
        y_scale: @Tensor<FP32x32>,
        y_zero_point: @Tensor<FP32x32>
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

    fn qlinear_concat(
        tensors: Span<Tensor<i8>>,
        scales: Span<Tensor<FP32x32>>,
        zero_points: Span<Tensor<FP32x32>>,
        y_scale: @Tensor<FP32x32>,
        y_zero_point: @Tensor<FP32x32>,
        axis: usize
    ) -> Tensor::<i8> {
        quantization::qlinear_concat::qlinear_concat(
            tensors,
            scales,
            zero_points,
            y_scale,
            y_zero_point,
            axis,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn qlinear_leakyrelu(
        self: @Tensor<i8>, a_scale: @Tensor<FP32x32>, a_zero_point: @Tensor<FP32x32>, alpha: FP32x32
    ) -> Tensor::<i8> {
        quantization::qlinear_leakyrelu::qlinear_leakyrelu(
            self,
            a_scale,
            a_zero_point,
            alpha,
            NumberTrait::new_unscaled(128, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn slice(
        self: @Tensor<FP32x32>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<FP32x32> {
        core_tensor::slice::<FP32x32>(self, starts, ends, axes, steps)
    }

    fn gather(self: @Tensor<FP32x32>, indices: Tensor<i32>, axis: Option<i32>) -> Tensor<FP32x32> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<FP32x32>) -> Tensor<usize> {
        core_tensor::nonzero(self)
    }

    fn squeeze(self: @Tensor<FP32x32>, axes: Option<Span<usize>>) -> Tensor<FP32x32> {
        core_tensor::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<FP32x32>, axes: Span<usize>) -> Tensor<FP32x32> {
        core_tensor::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::sign::sign(*self)
    }

    fn clip(self: @Tensor<FP32x32>, min: Option<FP32x32>, max: Option<FP32x32>) -> Tensor<FP32x32> {
        core_tensor::clip(self, min, max)
    }

    fn and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        core_tensor::identity(self)
    }

    fn where(self: @Tensor<FP32x32>, x: @Tensor<FP32x32>, y: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::where::where(self, x, y)
    }

    fn bitwise_and(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::bitwise_and::bitwise_and(self, other)
    }

    fn bitwise_xor(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::bitwise_xor::bitwise_xor(self, other)
    }

    fn bitwise_or(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::bitwise_or::bitwise_or(self, other)
    }

    fn round(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::round::round(*self)
    }

    fn trilu(self: @Tensor<FP32x32>, upper: bool, k: i64) -> Tensor<FP32x32> {
        linalg::trilu::trilu(self, upper, k)
    }

    fn reduce_l1(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_l1::reduce_l1(self, axis, keepdims)
    }

    fn scatter(
        self: @Tensor<FP32x32>,
        updates: Tensor<FP32x32>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<FP32x32> {
        math::scatter::scatter(self, updates, indices, axis, reduction)
    }

    fn array_feature_extractor(self: @Tensor<FP32x32>, indices: Tensor<usize>) -> Tensor<FP32x32> {
        ml::array_feature_extractor::array_feature_extractor(*self, indices)
    }

    fn binarizer(self: @Tensor<FP32x32>, threshold: Option<FP32x32>) -> Tensor<FP32x32> {
        math::binarizer::binarizer(*self, threshold)
    }

    fn reduce_sum_square(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_sum_square::reduce_sum_square(self, axis, keepdims)
    }

    fn reduce_l2(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_l2::reduce_l2(self, axis, keepdims)
    }

    fn not(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        panic(array!['not supported!'])
    }

    fn gather_elements(
        self: @Tensor<FP32x32>, indices: Tensor<i32>, axis: Option<i32>
    ) -> Tensor<FP32x32> {
        math::gather_elements::gather_elements(self, indices, axis)
    }

    fn shrink(
        self: Tensor<FP32x32>, bias: Option<FP32x32>, lambd: Option<FP32x32>
    ) -> Tensor<FP32x32> {
        math::shrink::shrink(self, bias, lambd)
    }

    fn reduce_mean(
        self: @Tensor<FP32x32>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP32x32> {
        math::reduce_mean::reduce_mean(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_min(
        self: @Tensor<FP32x32>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP32x32> {
        math::reduce_min::reduce_min(self, axes, keepdims, noop_with_empty_axes)
    }

    fn pow(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::pow::pow(self, other)
    }

    fn is_inf(
        self: @Tensor<FP32x32>, detect_negative: Option<u8>, detect_positive: Option<u8>
    ) -> Tensor<bool> {
        math::is_inf::is_inf(self, detect_negative, detect_positive)
    }

    fn is_nan(self: @Tensor<FP32x32>) -> Tensor<bool> {
        math::is_nan::is_nan(self)
    }

    fn gather_nd(
        self: @Tensor<FP32x32>, indices: Tensor<usize>, batch_dims: Option<usize>
    ) -> Tensor<FP32x32> {
        math::gather_nd::gather_nd(self, indices, batch_dims)
    }

    fn reduce_log_sum(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_log_sum::reduce_log_sum(self, axis, keepdims)
    }

    fn reduce_log_sum_exp(self: @Tensor<FP32x32>, axis: usize, keepdims: bool) -> Tensor<FP32x32> {
        math::reduce_log_sum_exp::reduce_log_sum_exp(self, axis, keepdims)
    }

    fn erf(self: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::erf::erf(*self)
    }

    fn unique(
        self: @Tensor<FP32x32>, axis: Option<usize>, sorted: Option<bool>
    ) -> (Tensor<FP32x32>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        manipulation::unique::unique(self, axis, sorted)
    }

    fn layer_normalization(
        self: @Tensor<FP32x32>,
        scale: @Tensor<FP32x32>,
        B: Option<@Tensor<FP32x32>>,
        axis: Option<i32>,
        epsilon: Option<FP32x32>,
        stash_type: Option<usize>,
    ) -> (Tensor<FP32x32>, Tensor<FP32x32>, Tensor<FP32x32>) {
        math::layer_normalization::layer_normalization(self, scale, B, axis, epsilon, stash_type)
    }

    fn resize(
        self: @Tensor<FP32x32>,
        roi: Option<Tensor<FP32x32>>,
        scales: Option<Span<FP32x32>>,
        sizes: Option<Span<usize>>,
        antialias: Option<usize>,
        axes: Option<Span<usize>>,
        coordinate_transformation_mode: Option<math::resize::TRANSFORMATION_MODE>,
        cubic_coeff_a: Option<FP32x32>,
        exclude_outside: Option<bool>,
        extrapolation_value: Option<FP32x32>,
        keep_aspect_ratio_policy: Option<math::resize::KEEP_ASPECT_RATIO_POLICY>,
        mode: Option<math::resize::MODE>,
        nearest_mode: Option<math::resize::NEAREST_MODE>,
    ) -> Tensor<FP32x32> {
        math::resize::resize(
            self,
            roi,
            scales,
            sizes,
            antialias,
            axes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy,
            mode,
            nearest_mode
        )
    }

    fn compress(
        self: @Tensor<FP32x32>, condition: Tensor<usize>, axis: Option<usize>
    ) -> Tensor<FP32x32> {
        math::compress::compress(self, condition, axis)
    }

    fn split(
        self: @Tensor<FP32x32>, axis: usize, num_outputs: Option<usize>, spl: Option<Tensor<usize>>
    ) -> Array<Tensor<FP32x32>> {
        manipulation::split::split(self, axis, num_outputs, spl)
    }

    fn random_uniform_like(
        tensor: @Tensor<FP32x32>, high: Option<FP32x32>, low: Option<FP32x32>, seed: Option<usize>
    ) -> Tensor<FP32x32> {
        math::random_uniform_like::random_uniform_like(*tensor, high, low, seed)
    }

    fn range(start: FP32x32, end: FP32x32, step: FP32x32) -> Tensor<FP32x32> {
        math::range::range(start, end, step)
    }

    fn hann_window(size: FP32x32, periodic: Option<usize>) -> Tensor<FP32x32> {
        panic(array!['not supported!'])
    }

    fn hamming_window(size: FP32x32, periodic: Option<usize>) -> Tensor<FP32x32> {
        panic(array!['not supported!'])
    }

    fn blackman_window(size: FP32x32, periodic: Option<usize>) -> Tensor<FP32x32> {
        panic(array!['not supported!'])
    }

    fn split_to_sequence(
        self: @Tensor<FP32x32>, axis: usize, keepdims: usize, split: Option<Tensor<usize>>
    ) -> Array<Tensor<FP32x32>> {
        manipulation::split_to_sequence::split_to_sequence(self, axis, keepdims, split)
    }

    fn reverse_sequence(
        self: @Tensor<FP32x32>,
        sequence_lens: Tensor<usize>,
        batch_axis: Option<usize>,
        time_axis: Option<usize>
    ) -> Tensor<FP32x32> {
        manipulation::reverse_sequence::reverse_sequence(self, sequence_lens, batch_axis, time_axis)
    }

    fn optional(self: @Tensor<FP32x32>) -> Option<Tensor<FP32x32>> {
        manipulation::optional::optional(self)
    }

    fn dynamic_quantize_linear(
        self: @Tensor<FP32x32>
    ) -> (Tensor::<u32>, Tensor::<FP32x32>, Tensor<FP32x32>) {
        quantization::dynamic_quantize_linear::dynamic_quantize_linear(
            self,
            NumberTrait::new_unscaled(0, false),
            NumberTrait::new_unscaled(255, false),
            NumberTrait::new_unscaled(0, false),
            NumberTrait::new_unscaled(1, false),
        )
    }

    fn scatter_nd(
        self: @Tensor<FP32x32>,
        updates: Tensor<FP32x32>,
        indices: Tensor<usize>,
        reduction: Option<usize>
    ) -> Tensor<FP32x32> {
        math::scatter_nd::scatter_nd(self, updates, indices, reduction)
    }

    fn label_encoder(
        self: @Tensor<FP32x32>,
        default_list: Option<Span<FP32x32>>,
        default_tensor: Option<Tensor<FP32x32>>,
        keys: Option<Span<FP32x32>>,
        keys_tensor: Option<Tensor<FP32x32>>,
        values: Option<Span<FP32x32>>,
        values_tensor: Option<Tensor<FP32x32>>
    ) -> Tensor<FP32x32> {
        ml::label_encoder::label_encoder(
            self, default_list, default_tensor, keys, keys_tensor, values, values_tensor
        )
    }
}

/// Implements addition for `Tensor<FP32x32>` using the `Add` trait.
impl FP32x32TensorAdd of Add<Tensor<FP32x32>> {
    /// Adds two `Tensor<FP32x32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP32x32>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FP32x32>` using the `Sub` trait.
impl FP32x32TensorSub of Sub<Tensor<FP32x32>> {
    /// Subtracts two `Tensor<FP32x32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP32x32>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FP32x32>` using the `Mul` trait.
impl FP32x32TensorMul of Mul<Tensor<FP32x32>> {
    /// Multiplies two `Tensor<FP32x32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP32x32>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FP32x32>` using the `Div` trait.
impl FP32x32TensorDiv of Div<Tensor<FP32x32>> {
    /// Divides two `Tensor<FP32x32>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP32x32>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<FP32x32>` using the `PartialEq` trait.
impl FP32x32TensorPartialEq of PartialEq<Tensor<FP32x32>> {
    fn eq(lhs: @Tensor<FP32x32>, rhs: @Tensor<FP32x32>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<FP32x32>, rhs: @Tensor<FP32x32>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl FP32x32TryIntoI8 of TryInto<FP32x32, i8> {
    fn try_into(self: FP32x32) -> Option<i8> {
        let number_felt: felt252 = (self.mag / ONE).into();
        let number_i8: i8 = number_felt.try_into().unwrap();
        if self.sign {
            return Option::Some(number_i8 * -1_i8);
        }
        Option::Some(number_i8)
    }
}

impl TensorI8IntoTensorFP32x32 of Into<Tensor<i8>, Tensor<FP32x32>> {
    fn into(self: Tensor<i8>) -> Tensor<FP32x32> {
        tensor_i8_to_tensor_fp32x32(@self)
    }
}

/// Implements partial ord for two `Tensor<FP32x32>` using `PartialOrd` trait.
impl FP32x32TensorPartialOrd of PartialOrd<Tensor<FP32x32>> {
    #[inline(always)]
    fn ge(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> bool {
        return SpanPartialOrd::ge(lhs.data, rhs.data);
    }

    #[inline(always)]
    fn gt(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> bool {
        return SpanPartialOrd::gt(lhs.data, rhs.data);
    }

    #[inline(always)]
    fn le(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> bool {
        return SpanPartialOrd::le(lhs.data, rhs.data);
    }

    #[inline(always)]
    fn lt(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> bool {
        return SpanPartialOrd::lt(lhs.data, rhs.data);
    }
}

// Internals
const PRECISION: u64 = 75497; // 0.009

fn relative_eq(lhs: @FP32x32, rhs: @FP32x32) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs.mag != 0 {
        (diff / *lhs).mag
    } else {
        diff.mag
    };

    rel_diff <= PRECISION
}

fn tensor_eq(mut lhs: Tensor<FP32x32>, mut rhs: Tensor<FP32x32>,) -> bool {
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
            is_eq = relative_eq(lhs.data.pop_front().unwrap(), rhs.data.pop_front().unwrap());
        };

    is_eq
}

fn tensor_i8_to_tensor_fp32x32(x: @Tensor<i8>) -> Tensor<FP32x32> {
    let mut result_data = ArrayTrait::<FP32x32>::new();
    let mut data = *x.data;

    while data.len() != 0 {
        result_data.append((*data.pop_front().unwrap()).into());
    };

    TensorTrait::new(*x.shape, result_data.span())
}
