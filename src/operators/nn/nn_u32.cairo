mod nn {
    use onnx_cairo::operators::tensor::core::Tensor;
    use onnx_cairo::operators::nn::functional::relu::relu_u32::relu_u32;

    /// Applies the rectified linear unit (ReLU) activation function element-wise to a given u32 tensor.
    ///
    /// The ReLU function is defined as f(x) = max(0, x), where x is the input element.
    ///
    /// # Arguments
    /// * `z` - A reference to an u32 tensor to which the ReLU function will be applied.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new u32 tensor with the same shape as the input tensor and the ReLU function
    ///   applied element-wise.
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        relu_u32(tensor)
    }
}
