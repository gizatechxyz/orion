use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::helpers::SpanPartialOrd;
use orion::operators::tensor::core::{
    new_tensor, constant_of_shape, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape,
    at_tensor,
};
use orion::operators::tensor::{math, linalg, quantization, core as core_tensor, ml, manipulation};
use orion::numbers::{NumberTrait, FP16x16, I8IntoFP16x16};
use orion::operators::tensor::implementations::{
    tensor_i8::I8Tensor, tensor_u32::U32Tensor, tensor_bool::BoolTensor
};
use orion::numbers::fixed_point::implementations::fp16x16::math::trig::PI;

use orion::numbers::fixed_point::implementations::fp16x16wide::core::FP16x16W;


impl FP16x16Tensor of TensorTrait<FP16x16> {
    fn new(shape: Span<usize>, data: Span<FP16x16>) -> Tensor<FP16x16> {
        new_tensor(shape, data)
    }

    fn constant_of_shape(shape: Span<usize>, value: FP16x16) -> Tensor<FP16x16> {
        constant_of_shape(shape, value)
    }

    fn add(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::add(@lhs, @rhs)
    }

    fn sub(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::sub(@lhs, @rhs)
    }

    fn mul(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::mul(@lhs, @rhs)
    }

    fn div(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::div(@lhs, @rhs)
    }

    fn at(self: @Tensor<FP16x16>, indices: Span<usize>) -> FP16x16 {
        *at_tensor(self, indices)
    }

    fn min_in_tensor(self: @Tensor<FP16x16>) -> FP16x16 {
        math::min_in_tensor::min_in_tensor::<FP16x16, u32>(*self.data)
    }

    fn min(tensors: Span<Tensor<FP16x16>>) -> Tensor<FP16x16> {
        math::min::min(tensors)
    }

    fn max_in_tensor(self: @Tensor<FP16x16>) -> FP16x16 {
        math::max_in_tensor::max_in_tensor(*self.data)
    }

    fn max(tensors: Span<Tensor<FP16x16>>) -> Tensor<FP16x16> {
        math::max::max(tensors)
    }

    fn stride(self: @Tensor<FP16x16>) -> Span<usize> {
        stride(*self.shape)
    }

    fn ravel_index(self: @Tensor<FP16x16>, indices: Span<usize>) -> usize {
        ravel_index(*self.shape, indices)
    }

    fn unravel_index(self: @Tensor<FP16x16>, index: usize) -> Span<usize> {
        unravel_index(index, *self.shape)
    }

    fn reshape(self: @Tensor<FP16x16>, target_shape: Span<i32>, allowzero: bool) -> Tensor<FP16x16> {
        reshape(self, target_shape, allowzero)
    }

    fn reduce_sum(
        self: @Tensor<FP16x16>,
        axes: Option<Span<i32>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP16x16> {
        math::reduce_sum::reduce_sum(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_prod(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        math::reduce_prod::reduce_prod(self, axis, keepdims)
    }

    fn argmax(
        self: @Tensor<FP16x16>, axis: i32, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<i32> {
        math::argmax::argmax(self, axis, keepdims, select_last_index)
    }

    fn argmin(
        self: @Tensor<FP16x16>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>
    ) -> Tensor<usize> {
        math::argmin::argmin(self, axis, keepdims, select_last_index)
    }

    fn transpose(self: @Tensor<FP16x16>, axes: Span<usize>) -> Tensor<FP16x16> {
        linalg::transpose::transpose(self, axes)
    }

    fn matmul(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        linalg::matmul::matmul(self, other)
    }

    fn exp(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::exp::exp(*self)
    }

    fn log(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::log::log(*self)
    }

    fn equal(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<usize> {
        math::equal::equal(self, other)
    }

    fn greater(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<usize> {
        math::greater::greater(self, other)
    }

    fn greater_equal(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<usize> {
        math::greater_equal::greater_equal(self, other)
    }

    fn less(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<i32> {
        math::less::less(self, other)
    }

    fn less_equal(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<i32> {
        math::less_equal::less_equal(self, other)
    }

    fn abs(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::abs::abs(*self)
    }

    fn neg(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::neg::neg(*self)
    }

    fn ceil(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::ceil::ceil(*self)
    }

    fn sin(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::sin::sin(*self)
    }

    fn cos(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::cos::cos(*self)
    }

    fn asin(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::asin::asin(*self)
    }

    fn cumsum(
        self: @Tensor<FP16x16>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
    ) -> Tensor<FP16x16> {
        math::cumsum::cumsum(self, axis, exclusive, reverse)
    }

    fn flatten(self: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        math::flatten::flatten(self, axis)
    }

    fn sinh(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::sinh::sinh(*self)
    }

    fn tanh(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::tanh::tanh(*self)
    }

    fn cosh(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::cosh::cosh(*self)
    }

    fn acosh(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::acosh::acosh(*self)
    }

    fn asinh(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::asinh::asinh(*self)
    }

    fn atan(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::atan::atan(*self)
    }

    fn xor(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<usize> {
        math::xor::xor(self, other)
    }

    fn or(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<usize> {
        math::or::or(self, other)
    }

    fn acos(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::acos::acos(*self)
    }

    fn onehot(
        self: @Tensor<FP16x16>, depth: usize, axis: Option<usize>, values: Span<usize>
    ) -> Tensor<FP16x16> {
        math::onehot::onehot(self, depth, axis, values)
    }

    fn sqrt(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::sqrt::sqrt(*self)
    }

    fn concat(tensors: Span<Tensor<FP16x16>>, axis: usize,) -> Tensor<FP16x16> {
        math::concat::concat(tensors, axis)
    }

    fn quantize_linear(
        self: @Tensor<FP16x16>, y_scale: @Tensor<FP16x16>, y_zero_point: @Tensor<FP16x16>
    ) -> Tensor::<i8> {
        quantization::quantize_linear::quantize_linear(
            self,
            y_scale,
            y_zero_point,
            NumberTrait::new_unscaled(127, true),
            NumberTrait::new_unscaled(127, false)
        )
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<FP16x16>, x_zero_point: @Tensor<FP16x16>
    ) -> Tensor::<FP16x16> {
        quantization::dequantize_linear::dequantize_linear(self, x_scale, x_zero_point)
    }

    fn qlinear_add(
        self: @Tensor<i8>,
        a_scale: @Tensor<FP16x16>,
        a_zero_point: @Tensor<FP16x16>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP16x16>,
        b_zero_point: @Tensor<FP16x16>,
        y_scale: @Tensor<FP16x16>,
        y_zero_point: @Tensor<FP16x16>
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
        a_scale: @Tensor<FP16x16>,
        a_zero_point: @Tensor<FP16x16>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP16x16>,
        b_zero_point: @Tensor<FP16x16>,
        y_scale: @Tensor<FP16x16>,
        y_zero_point: @Tensor<FP16x16>
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
        a_scale: @Tensor<FP16x16>,
        a_zero_point: @Tensor<FP16x16>,
        b: @Tensor<i8>,
        b_scale: @Tensor<FP16x16>,
        b_zero_point: @Tensor<FP16x16>,
        y_scale: @Tensor<FP16x16>,
        y_zero_point: @Tensor<FP16x16>
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
        scales: Span<Tensor<FP16x16>>,
        zero_points: Span<Tensor<FP16x16>>,
        y_scale: @Tensor<FP16x16>,
        y_zero_point: @Tensor<FP16x16>,
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
        self: @Tensor<i8>, a_scale: @Tensor<FP16x16>, a_zero_point: @Tensor<FP16x16>, alpha: FP16x16
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
        self: @Tensor<FP16x16>,
        starts: Span<usize>,
        ends: Span<usize>,
        axes: Option<Span<usize>>,
        steps: Option<Span<usize>>
    ) -> Tensor<FP16x16> {
        core_tensor::slice::<FP16x16>(self, starts, ends, axes, steps)
    }

    fn gather(
        self: @Tensor<FP16x16>, indices: Tensor<i32>, axis: Option<i32>
    ) -> Tensor<FP16x16> {
        math::gather::gather(self, indices, axis)
    }

    fn nonzero(self: @Tensor<FP16x16>) -> Tensor<usize> {
        core_tensor::nonzero(self)
    }

    fn squeeze(self: @Tensor<FP16x16>, axes: Option<Span<usize>>) -> Tensor<FP16x16> {
        core_tensor::squeeze(self, axes)
    }

    fn unsqueeze(self: @Tensor<FP16x16>, axes: Span<usize>) -> Tensor<FP16x16> {
        core_tensor::unsqueeze(self, axes)
    }

    fn sign(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::sign::sign(*self)
    }

    fn clip(self: @Tensor<FP16x16>, min: Option<FP16x16>, max: Option<FP16x16>) -> Tensor<FP16x16> {
        core_tensor::clip(self, min, max)
    }

    fn and(self: @Tensor<bool>, other: @Tensor<bool>) -> Tensor<bool> {
        math::and::and(self, other)
    }

    fn identity(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        core_tensor::identity(self)
    }

    fn where(self: @Tensor<FP16x16>, x: @Tensor<FP16x16>, y: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::where::where(self, x, y)
    }

    fn bitwise_and(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::bitwise_and::bitwise_and(self, other)
    }

    fn bitwise_xor(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::bitwise_xor::bitwise_xor(self, other)
    }

    fn bitwise_or(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::bitwise_or::bitwise_or(self, other)
    }

    fn round(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::round::round(*self)
    }

    fn reduce_l1(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        math::reduce_l1::reduce_l1(self, axis, keepdims)
    }

    fn array_feature_extractor(self: @Tensor<FP16x16>, indices: Tensor<usize>) -> Tensor<FP16x16> {
        ml::array_feature_extractor::array_feature_extractor(*self, indices)
    }

    fn binarizer(self: @Tensor<FP16x16>, threshold: Option<FP16x16>) -> Tensor<FP16x16> {
        math::binarizer::binarizer(*self, threshold)
    }

    fn reduce_sum_square(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        math::reduce_sum_square::reduce_sum_square(self, axis, keepdims)
    }

    fn reduce_l2(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        math::reduce_l2::reduce_l2(self, axis, keepdims)
    }

    fn trilu(self: @Tensor<FP16x16>, upper: bool, k: i64) -> Tensor<FP16x16> {
        linalg::trilu::trilu(self, upper, k)
    }

    fn scatter(
        self: @Tensor<FP16x16>,
        updates: Tensor<FP16x16>,
        indices: Tensor<usize>,
        axis: Option<usize>,
        reduction: Option<usize>
    ) -> Tensor<FP16x16> {
        math::scatter::scatter(self, updates, indices, axis, reduction)
    }

    fn not(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        panic(array!['not supported!'])
    }

    fn gather_elements(
        self: @Tensor<FP16x16>, indices: Tensor<i32>, axis: Option<i32>
    ) -> Tensor<FP16x16> {
        math::gather_elements::gather_elements(self, indices, axis)
    }

    fn shrink(
        self: Tensor<FP16x16>, bias: Option<FP16x16>, lambd: Option<FP16x16>
    ) -> Tensor<FP16x16> {
        math::shrink::shrink(self, bias, lambd)
    }

    fn reduce_mean(
        self: @Tensor<FP16x16>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP16x16> {
        math::reduce_mean::reduce_mean(self, axes, keepdims, noop_with_empty_axes)
    }

    fn reduce_min(
        self: @Tensor<FP16x16>,
        axes: Option<Span<usize>>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>
    ) -> Tensor<FP16x16> {
        math::reduce_min::reduce_min(self, axes, keepdims, noop_with_empty_axes)
    }

    fn pow(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::pow::pow(self, other)
    }

    fn is_inf(
        self: @Tensor<FP16x16>, detect_negative: Option<u8>, detect_positive: Option<u8>
    ) -> Tensor<bool> {
        math::is_inf::is_inf(self, detect_negative, detect_positive)
    }

    fn is_nan(self: @Tensor<FP16x16>) -> Tensor<bool> {
        math::is_nan::is_nan(self)
    }

    fn gather_nd(
        self: @Tensor<FP16x16>, indices: Tensor<usize>, batch_dims: Option<usize>
    ) -> Tensor<FP16x16> {
        math::gather_nd::gather_nd(self, indices, batch_dims)
    }

    fn reduce_log_sum(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        math::reduce_log_sum::reduce_log_sum(self, axis, keepdims)
    }

    fn reduce_log_sum_exp(self: @Tensor<FP16x16>, axis: usize, keepdims: bool) -> Tensor<FP16x16> {
        panic(array!['not supported!'])
    }


    fn erf(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::erf::erf(*self)
    }

    fn unique(
        self: @Tensor<FP16x16>, axis: Option<usize>, sorted: Option<bool>
    ) -> (Tensor<FP16x16>, Tensor<i32>, Tensor<i32>, Tensor<i32>) {
        manipulation::unique::unique(self, axis, sorted)
    }

    fn layer_normalization(
        self: @Tensor<FP16x16>,
        scale: @Tensor<FP16x16>,
        B: Option<@Tensor<FP16x16>>,
        axis: Option<i32>,
        epsilon: Option<FP16x16>,
        stash_type: Option<usize>,
    ) -> (Tensor<FP16x16>, Tensor<FP16x16>, Tensor<FP16x16>) {
        math::layer_normalization::layer_normalization(self, scale, B, axis, epsilon, stash_type)
    }

    fn resize(
        self: @Tensor<FP16x16>,
        roi: Option<Tensor<FP16x16>>,
        scales: Option<Span<FP16x16>>,
        sizes: Option<Span<usize>>,
        antialias: Option<usize>,
        axes: Option<Span<usize>>,
        coordinate_transformation_mode: Option<math::resize::TRANSFORMATION_MODE>,
        cubic_coeff_a: Option<FP16x16>,
        exclude_outside: Option<bool>,
        extrapolation_value: Option<FP16x16>,
        keep_aspect_ratio_policy: Option<math::resize::KEEP_ASPECT_RATIO_POLICY>,
        mode: Option<math::resize::MODE>,
        nearest_mode: Option<math::resize::NEAREST_MODE>,
    ) -> Tensor<FP16x16> {
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
        self: @Tensor<FP16x16>, condition: Tensor<usize>, axis: Option<usize>
    ) -> Tensor<FP16x16> {
        math::compress::compress(self, condition, axis)
    }

    fn split(
        self: @Tensor<FP16x16>, axis: usize, num_outputs: Option<usize>, spl: Option<Tensor<usize>>
    ) -> Array<Tensor<FP16x16>> {
        manipulation::split::split(self, axis, num_outputs, spl)
    }

    fn random_uniform_like(
        tensor: @Tensor<FP16x16>, high: Option<FP16x16>, low: Option<FP16x16>, seed: Option<usize>
    ) -> Tensor<FP16x16> {
        math::random_uniform_like::random_uniform_like(*tensor, high, low, seed)
    }

    fn range(start: FP16x16, end: FP16x16, step: FP16x16) -> Tensor<FP16x16> {
        math::range::range(start, end, step)
    }

    fn hann_window(size: FP16x16, periodic: Option<usize>) -> Tensor<FP16x16> {
        math::hann_window::hann_window(size, FP16x16 { mag: PI, sign: false }, periodic)
    }

    fn hamming_window(size: FP16x16, periodic: Option<usize>) -> Tensor<FP16x16> {
        math::hamming_window::hamming_window(size, FP16x16 { mag: PI, sign: false }, periodic)
    }

    fn blackman_window(size: FP16x16, periodic: Option<usize>) -> Tensor<FP16x16> {
        math::blackman_window::blackman_window(size, FP16x16 { mag: PI, sign: false }, periodic)
    }

    fn split_to_sequence(
        self: @Tensor<FP16x16>, axis: usize, keepdims: usize, split: Option<Tensor<usize>>
    ) -> Array<Tensor<FP16x16>> {
        manipulation::split_to_sequence::split_to_sequence(self, axis, keepdims, split)
    }

    fn reverse_sequence(
        self: @Tensor<FP16x16>,
        sequence_lens: Tensor<usize>,
        batch_axis: Option<usize>,
        time_axis: Option<usize>
    ) -> Tensor<FP16x16> {
        manipulation::reverse_sequence::reverse_sequence(self, sequence_lens, batch_axis, time_axis)
    }

    fn optional(self: @Tensor<FP16x16>) -> Option<Tensor<FP16x16>> {
        manipulation::optional::optional(self)
    }

    fn dynamic_quantize_linear(
        self: @Tensor<FP16x16>
    ) -> (Tensor::<u32>, Tensor::<FP16x16>, Tensor<FP16x16>) {
        quantization::dynamic_quantize_linear::dynamic_quantize_linear(
            self,
            NumberTrait::new_unscaled(0, false),
            NumberTrait::new_unscaled(255, false),
            NumberTrait::new_unscaled(0, false),
            NumberTrait::new_unscaled(1, false),
        )
    }

    fn scatter_nd(
        self: @Tensor<FP16x16>,
        updates: Tensor<FP16x16>,
        indices: Tensor<usize>,
        reduction: Option<usize>
    ) -> Tensor<FP16x16> {
        math::scatter_nd::scatter_nd(self, updates, indices, reduction)
    }

    fn label_encoder(
        self: @Tensor<FP16x16>,
        default_list: Option<Span<FP16x16>>,
        default_tensor: Option<Tensor<FP16x16>>,
        keys: Option<Span<FP16x16>>,
        keys_tensor: Option<Tensor<FP16x16>>,
        values: Option<Span<FP16x16>>,
        values_tensor: Option<Tensor<FP16x16>>
    ) -> Tensor<FP16x16> {
        ml::label_encoder::label_encoder(
            self, default_list, default_tensor, keys, keys_tensor, values, values_tensor
        )
    }

    fn mean(args: Span<Tensor<FP16x16>>) -> Tensor<FP16x16> {
        math::mean::mean(args)
    }
}

/// Implements addition for `Tensor<FP16x16>` using the `Add` trait.
impl FP16x16TensorAdd of Add<Tensor<FP16x16>> {
    /// Adds two `Tensor<FP16x16>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP16x16>` instance representing the result of the element-wise addition.
    fn add(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::add(@lhs, @rhs)
    }
}

/// Implements subtraction for `Tensor<FP16x16>` using the `Sub` trait.
impl FP16x16TensorSub of Sub<Tensor<FP16x16>> {
    /// Subtracts two `Tensor<FP16x16>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP16x16>` instance representing the result of the element-wise subtraction.
    fn sub(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::sub(@lhs, @rhs)
    }
}

/// Implements multiplication for `Tensor<FP16x16>` using the `Mul` trait.
impl FP16x16TensorMul of Mul<Tensor<FP16x16>> {
    /// Multiplies two `Tensor<FP16x16>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP16x16>` instance representing the result of the element-wise multiplication.
    fn mul(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::mul(@lhs, @rhs)
    }
}

/// Implements division for `Tensor<FP16x16>` using the `Div` trait.
impl FP16x16TensorDiv of Div<Tensor<FP16x16>> {
    /// Divides two `Tensor<FP16x16>` instances element-wise.
    ///
    /// # Arguments
    /// * `lhs` - The first tensor.
    /// * `rhs` - The second tensor.
    ///
    /// # Returns
    /// * A `Tensor<FP16x16>` instance representing the result of the element-wise division.
    fn div(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::div(@lhs, @rhs)
    }
}

/// Implements partial equal for two `Tensor<FP16x16>` using the `PartialEq` trait.
impl FP16x16TensorPartialEq of PartialEq<Tensor<FP16x16>> {
    fn eq(lhs: @Tensor<FP16x16>, rhs: @Tensor<FP16x16>) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor<FP16x16>, rhs: @Tensor<FP16x16>) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

impl TensorI8IntoTensorFP16x16 of Into<Tensor<i8>, Tensor<FP16x16>> {
    fn into(self: Tensor<i8>) -> Tensor<FP16x16> {
        tensor_i8_to_tensor_fp16x16(@self)
    }
}

/// Implements partial ord for two `Tensor<FP16x16>` using `PartialOrd` trait.
impl FP16x16TensorPartialOrd of PartialOrd<Tensor<FP16x16>> {
    #[inline(always)]
    fn ge(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> bool {
        SpanPartialOrd::ge(lhs.data, rhs.data)
    }

    #[inline(always)]
    fn gt(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> bool {
        SpanPartialOrd::gt(lhs.data, rhs.data)
    }

    #[inline(always)]
    fn le(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> bool {
        SpanPartialOrd::le(lhs.data, rhs.data)
    }

    #[inline(always)]
    fn lt(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> bool {
        SpanPartialOrd::lt(lhs.data, rhs.data)
    }
}

// Internals
const PRECISION: u32 = 589; // 0.009

fn relative_eq(lhs: @FP16x16, rhs: @FP16x16) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs.mag != 0 {
        (diff / *lhs).mag
    } else {
        diff.mag
    };

    rel_diff <= PRECISION
}


fn tensor_eq(mut lhs: Tensor<FP16x16>, mut rhs: Tensor<FP16x16>,) -> bool {
    let mut is_eq = true;

    while lhs.shape.len() != 0 && is_eq {
        is_eq = lhs.shape.pop_front().unwrap() == rhs.shape.pop_front().unwrap();
    };

    if !is_eq {
        return false;
    }

    while lhs.data.len() != 0 && is_eq {
        is_eq = relative_eq(lhs.data.pop_front().unwrap(), rhs.data.pop_front().unwrap());
    };

    is_eq
}

fn tensor_i8_to_tensor_fp16x16(x: @Tensor<i8>) -> Tensor<FP16x16> {
    let mut result_data = ArrayTrait::<FP16x16>::new();
    let mut data = *x.data;

    while data.len() != 0 {
        result_data.append((*data.pop_front().unwrap()).into());
    };

    TensorTrait::new(*x.shape, result_data.span())
}
