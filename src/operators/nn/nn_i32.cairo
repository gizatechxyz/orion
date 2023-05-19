mod NN {
    use onnx_cairo::operators::tensor::core::Tensor;
    use onnx_cairo::numbers::signed_integer::i32::i32;
    use onnx_cairo::operators::nn::functional::relu::relu_i32::relu_i32;
    use onnx_cairo::operators::nn::functional::leaky_relu::leaky_relu_i32::leaky_relu_i32;
    use onnx_cairo::operators::nn::functional::softmax::softmax_i32::softmax_i32;
    use onnx_cairo::numbers::fixed_point::core::FixedType;

    /// Applies the rectified linear unit (ReLU) activation function element-wise to a given i32 tensor.
    ///
    /// The ReLU function is defined as f(x) = max(0, x), where x is the input element.
    ///
    /// # Arguments
    /// * `tensor` - A reference to an i32 tensor to which the ReLU function will be applied.
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

    /// Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given i32 tensor.
    ///
    /// The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.
    ///
    /// # Arguments
    /// * `z` - A snapshot of a i32 tensor to which the Leaky ReLU function will be applied.
    /// * `alpha` - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.
    ///
    /// # Panics
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// # Returns
    /// * A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function
    ///   applied element-wise.
    fn leaky_relu(z: @Tensor<i32>, alpha: @FixedType) -> Tensor<FixedType> {
        leaky_relu_i32(z, alpha)
    }

    /// Calculates the softmax function for a tensor of i32 values along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A tensor of i32 values representing the input tensor.
    /// * `axis` - The axis along which to compute the softmax function.
    ///
    /// # Returns
    ///
    /// * A tensor of fixed point numbers representing the result of applying the softmax function 
    /// to the input tensor along the specified axis.
    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
        softmax_i32(tensor, axis)
    }
}
