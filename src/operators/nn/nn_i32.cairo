mod nn {
    use onnx_cairo::operators::tensor::core::Tensor;
    use onnx_cairo::operators::math::signed_integer::i32::i32;
    use onnx_cairo::operators::nn::functional::relu::relu_i32::relu_i32;

    /// Applies the rectified linear unit (ReLU) activation function element-wise to a given i32 tensor.
    ///
    /// The ReLU function is defined as f(x) = max(0, x), where x is the input element.
    ///
    /// # Arguments
    /// * `z` - A reference to an i32 tensor to which the ReLU function will be applied.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new i32 tensor with the same shape as the input tensor and the ReLU function
    ///   applied element-wise.
    fn relu(tensor: @Tensor<i32>) -> Tensor<i32> {
        relu_i32(tensor)
    }
}
