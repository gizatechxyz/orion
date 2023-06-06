use orion::operators::tensor::core::Tensor;
use orion::numbers::fixed_point::core::{FixedType};

/// Trait
///
/// relu - Applies the rectified linear unit function element-wise.
/// leaky_relu - Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise.
/// softmax - Computes softmax activations.
/// softsign - Applies the Softsign function element-wise.
/// softplus - Applies the Softplus function element-wise.
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
    /// * `threshold`(`T`) - A scalar that defines the threshold below which the Relu function returns 0.
    ///
    /// ## Returns
    /// 
    /// A `Tensor<T>` with the same shape as the input tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32;
    /// 
    /// fn relu_example() -> Tensor<u32> {
    ///     // We instantiate a 2D Tensor here and set threshold to 0.
    ///     // [[1,2],[-1,-2]]
    ///     let tensor = i32_tensor_2x2_helper();
    ///     let threshold = IntegerTrait::new(0, false);
    /// 		
    ///     // We can call `relu` function as follows.
    ///     return NNTrait::relu(@tensor, threshold);
    /// }
    /// >>> [[1,2],[0,0]]
    /// ```
    /// 
    fn relu(tensor: @Tensor<T>, threshold: T) -> Tensor<T>;
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
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_u32;
    /// 
    /// fn softmax_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `softmax` function as follows.
    ///     return NNTrait::softmax(@tensor, 1);
    /// }
    /// >>> [[2255697,6132911],[2255697,6132911]]
    ///     // The fixed point representation of
    ///     // [[0.2689, 0.7311],[0.2689, 0.7311]]
    /// ```
    ///
    fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
    /// # NNTrait::sigmoid
    ///
    /// ```rust 
    ///    fn sigmoid(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Applies the Sigmoid function to an n-dimensional input tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1].
    /// 
    /// $$
    /// \text{sigmoid}(x_i) = \frac{1}{1 + e^{-x_i}}
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_u32;
    /// 
    /// fn sigmoid_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `sigmoid` function as follows.
    ///     return NNTrait::sigmoid(@tensor);
    /// }
    /// >>> [[4194304,6132564],[7388661,7990771]]
    ///     // The fixed point representation of
    ///     // [[0.5, 0.7310586],[0.88079703, 0.95257413]]
    /// ```
    ///
    fn sigmoid(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// # NNTrait::softsign
    ///
    /// ```rust 
    ///    fn softsign(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Applies the Softsign function to an n-dimensional input Tensor such that the elements of the n-dimensional output Tensor lie in the range \[-1,1]. 
    /// 
    /// $$
    /// \text{softsign}(x_i) = \frac{x_i}{1 + |x_i|}
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_u32;
    /// 
    /// fn softsign_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `softsign` function as follows.
    ///     return NNTrait::softsign(@tensor);
    /// }
    /// >>> [[0,33554432],[44739242,50331648]]
    ///     // The fixed point representation of
    ///     // [[0, 0.5],[0.67, 0.75]]
    /// ```
    ///
    fn softsign(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// # NNTrait::softplus
    ///
    /// ```rust 
    ///    fn softplus(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// ```
    ///
    /// Applies the Softplus function to an n-dimensional input Tensor such that the elements of the n-dimensional output Tensor lie in the range \[-1,1].
    /// 
    /// $$
    /// \text{softplus{x_i} = \ln(1 + e^{x_i})
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_u32;
    /// 
    /// fn softplus_example() -> Tensor<FixedType> {
    ///     // We instantiate a 2D Tensor here.
    ///     // [[0,1],[2,3]]
    ///     let tensor = u32_tensor_2x2_helper();
    /// 		
    ///     // We can call `softplus` function as follows.
    ///     return NNTrait::softplus(@tensor);
    /// }
    /// >>> [[46516187,88131451],[142735719,204587229]]
    ///     // The fixed point representation of
    ///     // [[0.6931452, 1.31326096],[2.12692796, 3.04858728]]
    /// ```
    ///
    fn softplus(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// # NNTrait::linear
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
    /// * This function asserts that the input tensor `inputs` must be 1D, weights tensor must be 2D, and bias tensor must be 1D.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` representing the result of the linear transformation, possibly quantized.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32;
    /// 
    /// fn linear_layer_example() -> Tensor<u32> {
    ///     // We instantiate inputs here.
    ///     // inputs = [-71, 38, 62]
    ///     let inputs = i32_inputs_helper();
    /// 
    ///     // We instantiate weights here.
    ///     // weights = [[-8, 64, 40], [-33, -34, -20]]
    ///     let weights = i32_weights_helper();
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
    /// # NNTrait::leaky_relu
    /// 
    /// ```rust
    ///  fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType, threshold: T) -> Tensor<FixedType>
    /// ```
    ///
    /// Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.
    ///
    /// The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.
    ///
    /// ## Args
    /// * `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
    /// * `alpha`(`@FixedType`) - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.
    /// * `threshold`(`T`) - A scalar that defines the threshold below which the alpha value is applied.
    ///
    /// ## Panics
    ///
    /// * Panics if gas limit is exceeded during execution.
    ///
    /// ## Returns
    /// A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32;
    /// 
    /// fn leaky_relu_example() -> Tensor<u32> {
    ///     // We instantiate a 2D Tensor here, the alpha and set threshold to 0.
    ///     // [[1,2,-1],[-2,0,0]]
    ///     let tensor = i32_tensor_2x3_helper();
    ///     let alpha = Fixed::new(6710886_u128, false); // 0.1
    ///     let threshold = IntegerTrait::new(0, false);
    /// 		
    ///     // We can call `leaky_relu` function as follows.
    ///     return NNTrait::leaky_relu(@tensor, @alpha, threshold);
    /// }
    /// ```
    /// 
    fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType, threshold: T) -> Tensor<FixedType>;
}
