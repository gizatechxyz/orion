use orion::operators::tensor::core::Tensor;
use orion::numbers::fixed_point::core::{FixedType};

/// Trait
///
/// relu - Applies the rectified linear unit function element-wise.
/// leaky_relu - Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise.
/// sigmoid - Applies the Sigmoid function to an n-dimensional input tensor.
/// softmax - Computes softmax activations.
/// logsoftmax - Applies the natural log to Softmax function to an n-dimensional input Tensor.
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
    ///
    /// ## Returns
    /// 
    /// A `Tensor<T>` with the same shape as the input tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// 
    /// fn relu_example() -> Tensor<i32> {
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(1, true),
    ///             IntegerTrait::new(2, true),
    ///         ].span(),
    ///         extra: Option::None(())
    ///     );
    /// 
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
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn softmax_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return NNTrait::softmax(@tensor, 1);
    /// }
    /// >>> [[2255697,6132911],[2255697,6132911]]
    ///     // The fixed point representation of
    ///     // [[0.2689, 0.7311],[0.2689, 0.7311]]
    /// ```
    ///
    fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
    /// # NNTrait::logsoftmax
    ///
    /// ```rust 
    ///    fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>
    /// ```
    ///
    /// Applies the natural log to Softmax function to an n-dimensional input Tensor consisting of values in the range \[0,1].
    /// 
    /// $$
    /// \text{log softmax}(x_i) = \log(frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}})
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the natural lof softmax outputs.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn logsoftmax_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return NNTrait::logsoftmax(@tensor, 1);
    /// }
    ///     This will first generate the softmax output tensor
    /// >>> [[2255697,6132911],[2255697,6132911]]
    ///     // The fixed point representation of
    ///     // [[0.2689, 0.7311],[0.2689, 0.7311]]
    ///     
    ///     Applying the natural log to this tensor yields
    /// >>> 
    ///     // The fixed point representation of:
    ///     // [[-1.3134, -0.3132],[-1.3134, -0.3132]]
    /// ```
    ///
    fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
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
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn sigmoid_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
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
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn softsign_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return NNTrait::softsign(@tensor);
    /// }
    /// >>> [[0,4194304],[5592405,6291456]]
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
    /// \text{softplus}(x_i) = log({1 + e^{x_i}})
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
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn softplus_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return NNTrait::softplus(@tensor);
    /// }
    /// >>> [[5814540,11016447],[17841964,25573406]]
    ///     // The fixed point representation of
    ///     // [[0.6931452, 1.31326096],[2.12692796, 3.04858728]]
    /// ```
    ///
    fn softplus(tensor: @Tensor<T>) -> Tensor<FixedType>;
    /// # NNTrait::linear
    /// 
    /// ```rust
    /// fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>) -> Tensor<T>
    /// ```
    /// 
    /// Performs a linear transformation of the input tensor using the provided weights and bias.
    ///
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - A 1D tensor representing the input tensor.
    /// * `weights`(`@Tensor<T>`) - A 2D tensor representing the weights.
    /// * `bias`(`@Tensor<T>`) - A 1D tensor representing the bias.
    ///
    /// ## Panics
    /// 
    /// * This function asserts that the input tensor `inputs` must be 1D, weights tensor must be 2D, and bias tensor must be 1D.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` representing the result of the linear transformation.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType};
    /// 
    /// fn linear_example() -> Tensor<i32> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    /// 
    ///     // We instantiate inputs here.
    ///     let inputs = TensorTrait::<i32>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             IntegerTrait::new(71, true), IntegerTrait::new(38, false), IntegerTrait::new(62, false), 
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     // We instantiate weights here.
    ///     let weights = TensorTrait::<i32>::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             IntegerTrait::new(8, true),
    ///             IntegerTrait::new(64, false),
    ///             IntegerTrait::new(40, false),
    ///             IntegerTrait::new(33, true),
    ///             IntegerTrait::new(34, true),
    ///             IntegerTrait::new(20, true),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     // We instantiate bias here.
    ///     let bias = TensorTrait::<i32>::new(
    ///         shape: array![2].span(),
    ///         data: array![
    ///             IntegerTrait::new(61, false),
    ///             IntegerTrait::new(61, true),
    ///         ].span(),
    ///         extra: Option::Some(extra)
    ///     );
    /// 
    ///     return NNTrait::linear(inputs, weights, bias);
    /// }
    /// >>> [5541, -250]
    /// ````
    ///
    fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>) -> Tensor<T>;
    /// # NNTrait::leaky_relu
    /// 
    /// ```rust
    ///  fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType) -> Tensor<FixedType>
    /// ```
    ///
    /// Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.
    ///
    /// The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.
    ///
    /// ## Args
    /// * `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
    /// * `alpha`(`@FixedType`) - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.
    ///
    /// ## Returns
    /// A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    /// use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
    /// use orion::operators::nn::core::NNTrait;
    /// use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
    /// use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
    /// use orion::numbers::fixed_point::core::{FixedImpl, FixedType, FixedTrait};
    /// use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
    /// 
    /// fn leaky_relu_example() -> Tensor<FixedType> {
    ///     let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             IntegerTrait::new(1, false),
    ///             IntegerTrait::new(2, false),
    ///             IntegerTrait::new(1, true),
    ///             IntegerTrait::new(2, true),
    ///             IntegerTrait::new(0, false),
    ///             IntegerTrait::new(0, false),
    ///         ]
    ///             .span(),
    ///         extra: Option::Some(extra)
    ///     );
    ///     let alpha = FixedTrait::from_felt(838861); // 0.1
    /// 
    ///     return NNTrait::leaky_relu(@tensor, @alpha);
    /// }
    /// >>> [[8388608, 16777216, 838861], [1677722, 0, 0]]
    ///      // The fixed point representation of
    ///     [[1, 2, 0.1], [0.2, 0, 0]]
    /// ```
    /// 
    fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType) -> Tensor<FixedType>;
}
