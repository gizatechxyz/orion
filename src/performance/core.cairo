use orion::operators::tensor::core::Tensor;
use orion::numbers::fixed_point::core::FixedType;

/// Trait
///
/// quantize_linear - Quantizes a Tensor using linear quantization.
/// dequantize_linear - Dequantizes a Tensor using linear dequantization.
trait PerfomanceTrait<T, Q> {
    /// # performance.quantize_linear
    /// 
    /// ```rust
    /// fn quantize_linear(self: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<Q>;
    /// ```
    /// 
    /// Quantizes a Tensor using linear quantization.
    ///
    /// The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point 
    /// to compute the low precision / quantized tensor. The scale factor and zero point must have same shape, 
    /// and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
    /// The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`. For saturation, it saturates to `[-128, 127]`. 
    /// For (x / y_scale), it's rounding to the nearest even.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `y_scale`(`@Tensor<T>`) - Scale for doing quantization to get `y`.
    /// * `y_zero_point`(`@Tensor<T>`) - Zero point for doing quantization to get `y`.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the same shape as the input tensor, containing the quantized values.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::performance::core::PerfomanceTrait;
    /// use orion::performance::implementations::impl_performance_i32::Performance_i32_i8;
    /// 
    /// fn quantize_linear_example() -> Tensor<i8> {
    ///     // We instantiate a 1D Tensor here.
    ///     // [0, 2, 3, 1000, -254, -1000]
    ///     let x = i32_tensor_1D_helper();
    ///
    ///     // We instantiate the y_scale here.
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(IntegerTrait::new(2, false));
    ///     let extra = Option::<ExtraParams>::None(());
    ///     let y_scale = TensorTrait::new(shape.span(), data.span(), extra);
    /// 
    ///     // We instantiate the y_zero_point here.
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(IntegerTrait::new(1, false));
    ///     let extra = Option::<ExtraParams>::None(());
    ///     let y_zero_point = TensorTrait::new(shape.span(), data.span(), extra);
    /// 
    ///     // We can call `quantize_linear` function as follows.
    ///     return x.quantize_linear(@y_scale, @y_zero_point);
    /// }
    /// >>> [1, 2, 2, 127, -126, -128]
    /// ```
    ///
    fn quantize_linear(
        self: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>
    ) -> Tensor::<Q>;
    /// # performance.dequantize_linear
    /// 
    /// ```rust
    /// fn dequantize_linear(self: @Tensor<Q>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>) -> Tensor::<T>;
    /// ```
    /// 
    /// Dequantizes a Tensor using linear dequantization.
    ///
    /// The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute 
    /// the full precision tensor. The dequantization formula is y = (x - x_zero_point) * x_scale. x_scale and 
    /// x_zero_point must have same shape, and can be either a scalar for per-tensor / per layer quantization, 
    /// or a 1-D tensor for per-axis quantization.
    ///
    /// ## Args
    ///
    /// * `self`(`@Tensor<T>`) - The input tensor.
    /// * `x_scale`(`@Tensor<T>`) - Scale for input `x`.
    /// * `x_zero_point`(`@Tensor<T>`) - Zero point for input `x`.
    ///
    /// ## Returns
    ///
    /// A new `Tensor<T>` with the same shape as the input tensor, containing the dequantized values.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::performance::core::PerfomanceTrait;
    /// use orion::performance::implementations::impl_performance_i32::Performance_i32_i8;
    /// 
    /// fn quantize_linear_example() -> Tensor<i32> {
    ///     // We instantiate a 1D quantized Tensor here.
    ///     // [0, 3, 125, 127]
    ///     let x: Tensor<i8> = i32_tensor_1D_helper();
    ///
    ///     // We instantiate the x_scale here.
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(IntegerTrait::new(2, false));
    ///     let extra = Option::<ExtraParams>::None(());
    ///     let x_scale = TensorTrait::new(shape.span(), data.span(), extra);
    /// 
    ///     // We instantiate the x_zero_point here.
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(IntegerTrait::new(0, false));
    ///     let extra = Option::<ExtraParams>::None(());
    ///     let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);
    /// 
    ///     // We can call `dequantize_linear` function as follows.
    ///     return x.dequantize_linear(@x_scale, @x_zero_point);
    /// }
    /// >>> [0, 6, 250, 254]
    /// ```
    ///
    fn dequantize_linear(
        self: @Tensor<Q>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>
    ) -> Tensor::<T>;
}
