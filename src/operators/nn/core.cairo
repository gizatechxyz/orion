use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::numbers::fixed_point::core::FixedType;


/// Trait
///
/// relu - Applies the rectified linear unit function element-wise
/// softmax - Computes softmax activations.
/// linear - Performs a linear transformation of the input tensor using the provided weights and bias.  
trait NNTrait<T> {
    /// # NNTrait::relu
    ///
    /// ```rust 
    ///    fn relu(tensor: @Tensor<T>) -> Tensor<T>;
    /// ```
    /// 
    /// Applies the rectified linear unit function element-wise
    /// 
    /// $$
    /// ReLU(x)=(x)^+=max(0,x)
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    /// 
    /// A `Tensor<T>` with the same shape as the input tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use onnx_cairo::operators::nn::core::NNTrait;
    /// use onnx_cairo::operators::nn::implementations::impl_nn_i32;

    /// 
    /// fn relu_example() -> Tensor<u32> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[1,2],[-1,-2]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `relu` function as follows.
    ///     return NNTrait::relu(@tensor);
    /// }
    /// >>> [[1,2],[0,0]]
    /// ```
    /// 
    fn relu(tensor: @Tensor<T>) -> Tensor<T>;
    /// # NNTrait::softmax
    ///
    /// ```rust 
    ///    fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
    /// ```
    ///
    /// Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1] and sum to 1.
    /// 
    /// $$
    /// \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the softmax.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use onnx_cairo::operators::nn::core::NNTrait;
    /// use onnx_cairo::operators::nn::implementations::impl_nn_i32;
    /// 
    /// fn softmax_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `softmax` function as follows.
    ///     return NNTrait::softmax(@tensor, 1);
    /// }
    /// >>> [[18048353,49060510],[18048352,49060511]]
    ///     // The fixed point representation of
    ///     // [[0.2689, 0.7311],[0.2689, 0.7311]]
    /// ```
    ///
    fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
    /// # NNTrait::linear
    ///
    /// 
    /// ```rust
    /// fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>, quantized: bool) -> Tensor<T>
    /// ```
    /// 
    /// Performs a linear transformation of the input tensor using the provided weights and bias.
    ///
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - A 1D tensor representing the input tensor.
    /// * `weights`(`@Tensor<T>`) - A 2D tensor representing the weights.
    /// * `bias`(`@Tensor<T>`) - A 1D tensor representing the bias.
    /// * `quantized`(`bool`) - A boolean flag indicating whether or not to quantize the result.
    ///
    /// ## Panics
    /// 
    /// This function asserts that the input tensor `inputs` must be 1D, weights tensor must be 2D, and bias tensor must be 1D.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` representing the result of the linear transformation, possibly quantized.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use onnx_cairo::operators::nn::core::NNTrait;
    /// use onnx_cairo::operators::nn::implementations::impl_nn_i32;
    /// 
    /// fn linear_layer_example() -> Tensor<u32> {
    ///     // We instantiate inputs here.
    ///     // inputs = [-71, 38, 62]
    ///     let inputs = u32_inputs_helper();
    /// 
    ///     // We instantiate weights here.
    ///     // weights = [[-8, 64, 40], [-33, -34, -20]]
    ///     let weights = u32_weights_helper();
    /// 
    ///     // We instantiate bias here.
    ///     // weights = [61, -71]
    ///     let weights = u32_bias_helper();
    /// 
    ///     // We can call `linear` function as follows.
    ///     return NNTrait::linear(inputs, weights, bias, true);
    /// }
    /// >>> [127, -6]
    /// ````
    ///
    fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>, quantized: bool) -> Tensor<T>;
}
