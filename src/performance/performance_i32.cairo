mod performance {
    use onnx_cairo::operators::math::signed_integer::i32::i32;
    use onnx_cairo::operators::tensor::core::Tensor;
    use onnx_cairo::performance::functional::quantization::quant_i32::quantize_tensor;

    /// Quantizes an i32 tensor using symmetric quantization.
    ///
    /// # Arguments
    /// * `tensor` - A reference to an i32 tensor to be quantized.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new i32 tensor with the same shape as the input tensor, containing the quantized values.
    fn quantize_linear(self: @Tensor<i32>) -> Tensor<i32> {
        quantize_tensor(self)
    }
}
