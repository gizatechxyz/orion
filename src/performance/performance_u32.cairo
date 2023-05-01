mod performance {
    use onnx_cairo::operators::tensor::core::Tensor;
    use onnx_cairo::performance::functional::quantization::quant_u32::quantize_tensor;

    /// Quantizes an u32 tensor using symmetric quantization.
    ///
    /// # Arguments
    /// * `tensor` - A reference to an u32 tensor to be quantized.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new u32 tensor with the same shape as the input tensor, containing the quantized values.
    fn quantize_linear(self: @Tensor<u32>) -> Tensor<u32> {
        quantize_tensor(self)
    }
}
