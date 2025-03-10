use orion::operators::tensor::core::Tensor;

/// Trait
///
/// relu - Applies the rectified linear unit function element-wise.
/// leaky_relu - Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise.
/// sigmoid - Applies the Sigmoid function to an n-dimensional input tensor.
/// softmax - Computes softmax activations.
/// softmax_zero - Computes softmax zero.
/// logsoftmax - Applies the natural log to Softmax function to an n-dimensional input Tensor.
/// softsign - Applies the Softsign function element-wise.
/// softplus - Applies the Softplus function element-wise.
/// linear - Performs a linear transformation of the input tensor using the provided weights and bias.
/// hard_sigmoid - Applies the Hard Sigmoid function to an n-dimensional input tensor.
/// thresholded_relu - Performs the thresholded relu activation function element-wise.
/// gemm - Performs General Matrix multiplication.
/// grid_sample - Computes the grid sample of the input tensor and input grid.
/// col2im - Rearranges column blocks back into a multidimensional image
/// conv_transpose - Performs the convolution transpose of the input data tensor and weight tensor.
/// conv - Performs the convolution of the input data tensor and weight tensor.
/// hard_swish - Applies the Hard Swish function to an n-dimensional input tensor.
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
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
    /// use orion::operators::nn::{NNTrait, I32NN};
    /// 
    /// fn relu_example() -> Tensor<i32> {
    ///     let tensor = TensorTrait::<i32>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![1, 2, -1, -2].span(),
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
    ///    fn softmax(tensor: @Tensor<T>, axis: Option<i32>) -> Tensor<T>;
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
    /// * `axis`(`Option<i32>`) - Describes the dimension Softmax will be performed on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn softmax_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::softmax(@tensor, Option::Some(1));
    /// }
    /// >>> [[2255697,6132911],[2255697,6132911]]
    ///     // The fixed point representation of
    ///     // [[0.2689, 0.7311],[0.2689, 0.7311]]
    /// ```
    ///
    fn softmax(tensor: @Tensor<T>, axis: Option<i32>) -> Tensor<T>;
    /// # NNTrait::softmax_zero
    ///
    /// ```rust 
    ///    fn softmax_zero(tensor: @Tensor<T>, axis: usize) -> Tensor<T>;
    /// ```
    ///
    /// Applies the Softmax zero function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range \[0,1] and sum to 1 while keeping the zero elements to zero.
    /// 
    /// The softmax zero on the set $\mathbf{x} = (x_1, ..., x_n)$ is given by :
    /// 
    /// $$
    /// \text{softmax zero}(x_i) = \begin{cases}
    /// 0 & \qquad x_i = 0 \\
    /// \frac{e^{x_i}}{ \sum_{x \in {S}} e^{x}} & \qquad \text{otherwise} 
    /// \end{cases}
    /// $$
    /// where $S$ in a subset of $\mathbf{x}$ given by 
    /// 
    /// $$
    ///  \  S = \{ (x_1, \ldots, x_k) \mid 1 \leq k \leq n, x_j \neq 0 \text{ for } 1 \leq j \leq k \}
    /// $$
    ///
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    /// * `axis`(`usize`) - The axis along which to compute the softmax zero.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// use core::debug::PrintTrait;
    /// 
    /// fn softmax_zero_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(8388608, false),
    ///             FixedTrait::new(16777216, false),
    ///             FixedTrait::new(25165824, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::softmax_zero(@tensor, 1);
    /// }
    /// >>> [[0,0x800000],[2256043,6132564]]
    ///     // The fixed point representation of
    ///     // [[0, 1],[0.2689, 0.7311]]
    /// ```
    ///
    fn softmax_zero(tensor: @Tensor<T>, axis: usize) -> Tensor<T>;
    /// # NNTrait::logsoftmax
    ///
    /// ```rust 
    ///    fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<T>
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn logsoftmax_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false),
    ///         ]
    ///             .span(),
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
    fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<T>;
    /// # NNTrait::sigmoid
    ///
    /// ```rust 
    ///    fn sigmoid(tensor: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn sigmoid_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::sigmoid(@tensor);
    /// }
    /// >>> [[4194304,6132564],[7388661,7990771]]
    ///     // The fixed point representation of
    ///     // [[0.5, 0.7310586],[0.88079703, 0.95257413]]
    /// ```
    ///
    fn sigmoid(tensor: @Tensor<T>) -> Tensor<T>;
    /// # NNTrait::softsign
    ///
    /// ```rust 
    ///    fn softsign(tensor: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn softsign_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::softsign(@tensor);
    /// }
    /// >>> [[0,4194304],[5592405,6291456]]
    ///     // The fixed point representation of
    ///     // [[0, 0.5],[0.67, 0.75]]
    /// ```
    ///
    fn softsign(tensor: @Tensor<T>) -> Tensor<T>;
    /// # NNTrait::softplus
    ///
    /// ```rust 
    ///    fn softplus(tensor: @Tensor<T>) -> Tensor<T>;
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
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn softplus_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(3, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::softplus(@tensor);
    /// }
    /// >>> [[5814540,11016447],[17841964,25573406]]
    ///     // The fixed point representation of
    ///     // [[0.6931452, 1.31326096],[2.12692796, 3.04858728]]
    /// ```
    ///
    fn softplus(tensor: @Tensor<T>) -> Tensor<T>;
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
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
    /// use orion::operators::nn::{NNTrait, I32NN};
    /// 
    /// fn linear_example() -> Tensor<i32> {
    ///     // We instantiate inputs here.
    ///     let inputs = TensorTrait::<i32>::new(
    ///         shape: array![3].span(),
    ///         data: array![
    ///             -71, 38, 62,
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     // We instantiate weights here.
    ///     let weights = TensorTrait::<i32>::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             -8,
    ///             64,
    ///             40,
    ///             -33,
    ///             -34,
    ///             -20,
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     // We instantiate bias here.
    ///     let bias = TensorTrait::<i32>::new(
    ///         shape: array![2].span(),
    ///         data: array![61, -61].span(),
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
    ///  fn leaky_relu(inputs: @Tensor<T>, alpha: @T) -> Tensor<T>
    /// ```
    ///
    /// Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.
    ///
    /// The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.
    ///
    /// ## Args
    /// * `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
    /// * `alpha`(`@T`) - A snapshot of a fixed point scalar that defines the alpha value of the Leaky ReLU function.
    ///
    /// ## Returns
    /// A new fixed point tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn leaky_relu_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 3].span(),
    ///         data: array![
    ///             FixedTrait::new(1, false),
    ///             FixedTrait::new(2, false),
    ///             FixedTrait::new(1, true),
    ///             FixedTrait::new(2, true),
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(0, false),
    ///         ]
    ///             .span(),
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
    fn leaky_relu(inputs: @Tensor<T>, alpha: @T) -> Tensor<T>;
    /// # NNTrait::hard_sigmoid
    ///
    /// ```rust 
    ///    fn hard_sigmoid(tensor: @Tensor<T>, alpha: @T, beta: @T) -> Tensor<T>;
    /// ```
    ///
    /// Applies the HardSigmoid function to an n-dimensional input tensor.
    /// 
    /// $$
    /// \text{HardSigmoid}(x_i) = \text{max}(0, \text{min}(alpha * x + beta, 1))
    /// $$
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    /// * `alpha`(`@T`) - value of alpha.
    /// * `beta`(`@T`) - value of beta.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn hard_sigmoid_example() -> Tensor<FP16x16> {
    ///     let tensor = TensorTrait::<FP16x16>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(13107, false),
    ///             FixedTrait::new(32768, false),
    ///             FixedTrait::new(65536, false),
    ///         ]
    ///             .span(),
    ///     );
    ///     let alpha = FixedTrait::new(13107, false);
    ///     let beta = FixedTrait::new(32768, false);
    /// 
    ///     return NNTrait::hard_sigmoid(@tensor, @alpha, @beta);
    /// }
    /// >>> [[32768, 35389],[39321, 45875]]
    /// ```
    ///
    fn hard_sigmoid(tensor: @Tensor<T>, alpha: @T, beta: @T) -> Tensor<T>;
    /// # NNTrait::hard_swish
    ///
    /// ```rust 
    ///    fn hard_swish(tensor: @Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Applies the HardSwish function to an n-dimensional input tensor.
    /// 
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor.
    ///
    /// ## Returns
    ///
    /// A Tensor of fixed point numbers with the same shape than the input Tensor.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn hard_swish_example() -> Tensor<FP16x16> {
    ///     let tensor = TensorTrait::<FP16x16>::new(
    ///         shape: array![2, 2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(87989, true),
    ///             FixedTrait::new(13107, false),
    ///             FixedTrait::new(32768, true),
    ///             FixedTrait::new(65536, false),
    ///             FixedTrait::new(89090, true),
    ///             FixedTrait::new(13107, false),
    ///             FixedTrait::new(38988, true),
    ///             FixedTrait::new(78990, false),
    ///         ]
    ///             .span(),
    ///     );
    /// 
    ///     return NNTrait::hard_swish(@tensor);
    /// }
    /// >>> [[[-0.37089539,  0.10665894],
    ///       [-0.20832825,  0.66665649]],
    /// 
    ///      [[-0.37171936,  0.10665894],
    ///       [-0.23846436,  0.84474182]]]
    /// ```
    ///
    fn hard_swish(tensor: @Tensor<T>) -> Tensor<T>;
    /// # NNTrait::thresholded_relu
    /// 
    /// ```rust
    ///  fn thresholded_relu(tensor: @Tensor<T>, alpha: @T) -> Tensor<T>
    /// ```
    ///
    /// Applies the thresholded rectified linear unit (Thresholded ReLU) activation function element-wise to a given tensor.
    ///
    /// The Thresholded ReLU function is defined as f(x) = x if x > alpha, f(x) = 0 otherwise, where x is the input element.
    ///
    /// ## Args
    /// * `tensor`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
    /// * `alpha`(`@T`) - A snapshot of a fixed point scalar that defines the alpha value of the Thresholded ReLU function.
    ///
    /// ## Returns
    /// A new fixed point tensor with the same shape as the input tensor and the Thresholded ReLU function applied element-wise.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
    /// use orion::operators::nn::{NNTrait, FP8x23NN};
    /// use orion::numbers::{FP8x23, FixedTrait};
    /// 
    /// fn thresholded_relu_example() -> Tensor<FP8x23> {
    ///     let tensor = TensorTrait::<FP8x23>::new(
    ///         shape: array![2, 2].span(),
    ///         data: array![
    ///             FixedTrait::new(0, false),
    ///             FixedTrait::new(256, false),
    ///             FixedTrait::new(512, false),
    ///             FixedTrait::new(257, false),
    ///         ]
    ///             .span(),
    ///     );
    ///     let alpha = FixedTrait::from_felt(256); // 1.0
    /// 
    ///     return NNTrait::leaky_relu(@tensor, @alpha);
    /// }
    /// >>> [[0, 0], [512, 257]]
    /// ```
    /// 
    fn thresholded_relu(tensor: @Tensor<T>, alpha: @T) -> Tensor<T>;
    /// # NNTrait::space_to_depth
    /// 
    /// ```rust
    ///     fn space_to_depth(tensor: @Tensor<T>, blocksize: usize) -> Tensor<T>;
    /// ```
    ///
    /// SpaceToDepth rearranges blocks of spatial data into depth. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the depth dimension.
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
    /// * `blocksize`(`usize`) - The size of the blocks to move along [blocksize, blocksize].
    ///
    /// ## Returns
    /// 
    /// A `Tensor<T>` of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// use orion::operators::tensor::{TensorTrait, Tensor};
    /// use orion::operators::tensor::{I8Tensor, I8TensorAdd};
    /// use orion::numbers::NumberTrait;
    /// use orion::operators::nn::NNTrait;
    /// use orion::operators::nn::I8NN;
    /// use orion::numbers::FixedTrait;
    ///
    /// fn space_to_depth_example() -> Tensor<i8> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(2);
    ///     shape.append(2);
    ///     shape.append(4);
    ///
    ///     let mut data = ArrayTrait::new();
    ///     data.append(-3);
    ///     data.append(0);
    ///     data.append(0);
    ///     data.append(0);
    ///     data.append(-1);
    ///     data.append(1);
    ///     data.append(-2);
    ///     data.append(-3);
    ///     data.append(2);
    ///     data.append(-2);
    ///     data.append(-3);
    ///     data.append(-3);
    ///     data.append(-1);
    ///     data.append(0);
    ///     data.append(1);
    ///     data.append(-3);
    ///     let tensor = TensorTrait::new(shape.span(), data.span());
    ///     return NNTrait::space_to_depth(@tensor, 2);
    /// }
    /// >>> [[[[-3, 0]], [[2, -3]], [[0, 0]], [[-2, -3]], [[-1, -2]], [[-1, 1]], [[1, -3]], [[0, -3]]]]
    /// ```
    ///
    fn space_to_depth(tensor: @Tensor<T>, blocksize: usize) -> Tensor<T>;
    /// # NNTrait::depth_to_space
    /// 
    /// ```rust
    ///     fn depth_to_space(tensor: @Tensor<T>, blocksize: usize) -> Tensor<T>;
    /// ```
    ///
    /// DepthToSpace rearranges (permutes) data from depth into blocks of spatial data. This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of the input tensor where values from the depth dimension are moved in spatial blocks to the height and width dimensions. By default, mode = DCR. In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the following order: depth, column, and then row. 
    /// 
    /// ## Args
    ///
    /// * `tensor`(`@Tensor<T>`) - The input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
    /// * `blocksize`(`usize`) - The size of the blocks to move along [blocksize, blocksize].
    /// * `mode`(felt252) - DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.
    ///
    /// ## Returns
    /// 
    /// A `Tensor<T>` of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// use orion::operators::tensor::{TensorTrait, Tensor};
    /// use orion::operators::tensor::{I8Tensor, I8TensorAdd};
    /// use orion::numbers::NumberTrait;
    /// use orion::operators::nn::NNTrait;
    /// use orion::operators::nn::I8NN;
    /// use orion::numbers::FixedTrait;
    ///
    /// fn depth_to_space_example() -> Tensor<i8> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(4);
    ///     shape.append(2);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(-2);
    ///     data.append(0);
    ///     data.append(-1);
    ///     data.append(0);
    ///     data.append(0);
    ///     data.append(-3);
    ///     data.append(2);
    ///     data.append(1);
    ///     data.append(-2);
    ///     data.append(-2);
    ///     data.append(0);
    ///     data.append(-2);
    ///     data.append(-1);
    ///     data.append(-1);
    ///     data.append(2);
    ///     data.append(2);
    ///     let tensor = TensorTrait::new(shape.span(), data.span());
    ///     return NNTrait::depth_to_space(@tensor, 2, 'DCR');
    /// }
    /// >>> [[[[-2, 0, 0, -3], [-2, -1, -2, -1], [-1, 2, 0, 1], [0, 2, -2, 2]]]]
    /// ```
    ///
    fn depth_to_space(tensor: @Tensor<T>, blocksize: usize, mode: felt252) -> Tensor<T>;
    /// # NNTrait::gemm
    /// 
    /// ```rust
    ///     fn gemm(
    ///         A: Tensor<T>,
    ///         B: Tensor<T>,
    ///         C: Option<Tensor<T>>,
    ///         alpha: Option<T>,
    ///         beta: Option<T>,
    ///         transA: bool,
    ///         transB: bool
    ///     ) -> Tensor<T>;
    /// ```
    /// 
    /// Performs General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    ///
    /// * A' = transpose(A) if transA else A
    /// * B' = transpose(B) if transB else B
    ///
    /// Compute `Y = alpha * A' * B' + beta * C`, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
    /// `A` will be transposed before doing the computation if attribute `transA` is `true`, same for `B` and `transB`.
    /// 
    /// ## Args
    ///
    /// * `A`(`Tensor<T>`) - Input tensor A. The shape of `A` should be (M, K) if `transA` is `false`, or (K, M) if `transA` is `true`.
    /// * `B`(`Tensor<T>`) - Input tensor B. The shape of `B` should be (K, N) if `transB` is `false`, or (N, K) if `transB` is `true`.
    /// * `C`(`Option<Tensor<T>>`) - Optional input tensor C. The shape of C should be unidirectional broadcastable to (M, N). 
    /// * `alpha`(`Option<T>`) - Optional scalar multiplier for the product of input tensors `A * B`.
    /// * `beta`(`Option<T>`) - Optional scalar multiplier for input tensor `C`.
    /// * `transA`(`bool`) - Whether `A` should be transposed.
    /// * `transB`(`bool`) - Whether `B` should be transposed.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` of shape (M, N).
    ///
    /// ## Examples
    ///
    /// ```rust
    ///     mod input_0;
    ///     mod input_1;
    ///     mod input_2;
    ///     
    ///     use orion::operators::nn::NNTrait;
    ///     use orion::numbers::FixedTrait;
    ///     use orion::operators::nn::FP16x16NN;
    ///     use orion::operators::tensor::FP16x16TensorPartialEq;
    ///
    ///   fn gemm_all_attributes_example() -> Tensor<FP16x16> {
    ///       let input_0 = input_0::input_0(); // shape [4;3]
    ///       let input_1 = input_1::input_1(); // shape [5;4]
    ///       let input_2 = input_2::input_2(); // shape [1;5]
    ///
    ///       let y = NNTrait::gemm(
    ///           input_0,
    ///           input_1,
    ///           Option::Some(input_2),
    ///           Option::Some(FixedTrait::new(16384, false)), // 0.25
    ///           Option::Some(FixedTrait::new(22938, false)), // 0.35
    ///           true,
    ///           true
    ///       );
    ///
    ///       return y;
    ///   } 
    ///  >>> tensor of shape [3;5]
    /// ````
    ///
    fn gemm(
        A: Tensor<T>,
        B: Tensor<T>,
        C: Option<Tensor<T>>,
        alpha: Option<T>,
        beta: Option<T>,
        transA: bool,
        transB: bool
    ) -> Tensor<T>;
    ///
    /// # NNTrait::conv
    /// 
    /// ```rust
    ///     conv(
    ///         X: @Tensor<T>,
    ///         W: @Tensor<T>,
    ///         B: Option<Span<T>>,
    ///         auto_pad: Option<AUTO_PAD>,
    ///         dilations: Option<Span<usize>>,
    ///         group: Option<usize>,
    ///         kernel_shape: Option<Span<usize>>,
    ///         pads: Option<Span<usize>>,
    ///         strides: Option<Span<usize>>,
    ///     ) -> Tensor<T>
    /// ```
    /// 
    /// The convolution operator consumes an input tensor and a filter (input weight tensor), and computes the output.
    ///
    /// ## Args
    ///
    /// * `X`(`@Tensor<T>`) - Input data tensor, has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W if 2D, otherwise the size is (N x C x D1 x D2 ... x Dn).
    /// * `W`(`@Tensor<T>`) - The weight tensor, has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps if 2D, for more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn).
    /// * `B`(`Option<@Tensor<T>>`) - Optional 1D bias to be added to the convolution, has size of M.
    /// * `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
    /// * `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
    /// * `group`(`Option<usize>`) - Default is 1, number of groups input channels and output channels are divided into.
    /// * `kernel_shape`(`Option<Span<usize>>`) - The shape of the convolution kernel. If not present, should be inferred from input W.
    /// * `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
    /// * `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` that contains the result of the convolution.
    ///
    /// ## Examples
    ///     
    /// ```rust
    /// use orion::operators::nn::NNTrait;
    /// use orion::numbers::FixedTrait;
    /// use orion::operators::nn::FP16x16NN;
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
    /// 
    /// 
    /// fn example_conv() -> Tensor<FP16x16> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(1);
    ///     shape.append(3);
    ///     shape.append(3);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     let W = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(1);
    ///     shape.append(5);
    ///     shape.append(5);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 0, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 131072, sign: false });
    ///     data.append(FP16x16 { mag: 196608, sign: false });
    ///     data.append(FP16x16 { mag: 262144, sign: false });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     data.append(FP16x16 { mag: 393216, sign: false });
    ///     data.append(FP16x16 { mag: 458752, sign: false });
    ///     data.append(FP16x16 { mag: 524288, sign: false });
    ///     data.append(FP16x16 { mag: 589824, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 720896, sign: false });
    ///     data.append(FP16x16 { mag: 786432, sign: false });
    ///     data.append(FP16x16 { mag: 851968, sign: false });
    ///     data.append(FP16x16 { mag: 917504, sign: false });
    ///     data.append(FP16x16 { mag: 983040, sign: false });
    ///     data.append(FP16x16 { mag: 1048576, sign: false });
    ///     data.append(FP16x16 { mag: 1114112, sign: false });
    ///     data.append(FP16x16 { mag: 1179648, sign: false });
    ///     data.append(FP16x16 { mag: 1245184, sign: false });
    ///     data.append(FP16x16 { mag: 1310720, sign: false });
    ///     data.append(FP16x16 { mag: 1376256, sign: false });
    ///     data.append(FP16x16 { mag: 1441792, sign: false });
    ///     data.append(FP16x16 { mag: 1507328, sign: false });
    ///     data.append(FP16x16 { mag: 1572864, sign: false });
    ///     let mut X = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     return NNTrait::conv(
    ///         @X,
    ///         @W,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::Some(array![3, 3].span()),
    ///         Option::Some(array![1, 1, 1, 1].span()),
    ///         Option::None,
    ///     );
    /// }
    ///
    /// >>> [
    ///         [
    ///             [
    ///                 [12.0, 21.0, 27.0, 33.0, 24.0],  
    ///                 [33.0, 54.0, 63.0, 72.0, 51.0],
    ///                 [63.0, 99.0, 108.0, 117.0, 81.0],
    ///                 [93.0, 144.0, 153.0, 162.0, 111.0],
    ///                 [72.0, 111.0, 117.0, 123.0, 84.0],
    ///             ]
    ///         ]
    ///     ]
    ///
    /// ````
    ///
    fn conv(
        X: @Tensor<T>,
        W: @Tensor<T>,
        B: Option<Span<T>>,
        auto_pad: Option<orion::operators::nn::functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<T>;
    /// # NNTrait::conv_transpose
    /// 
    /// ```rust
    ///     conv_transpose(
    ///     X: @Tensor<T>,
    ///     W: @Tensor<T>,
    ///     B: Option<@Tensor<T>>,
    ///     auto_pad: Option<AUTO_PAD>,
    ///     dilations: Option<Span<usize>>,
    ///     group: Option<usize>,
    ///     kernel_shape: Option<Span<usize>>,
    ///     output_padding: Option<Span<usize>>,
    ///     output_shape: Option<Span<usize>>,
    ///     pads: Option<Span<usize>>,
    ///     strides: Option<Span<usize>>,
    /// ) -> Tensor<T>
    /// ```
    /// 
    /// The convolution transpose operator consumes an input tensor and a input weight tensor, and computes the output.
    ///
    /// ## Args
    ///
    /// * `X`(`@Tensor<T>`) - Input data tensor, has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W if 2D, otherwise the size is (N x C x D1 x D2 ... x Dn).
    /// * `W`(`@Tensor<T>`) - The weight tensor, has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps if 2D, for more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn).
    /// * `B`(`Option<@Tensor<T>>`) - Optional 1D bias to be added to the convolution, has size of M.
    /// * `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`.
    /// * `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
    /// * `group`(`Option<usize>`) - Default is 1, number of groups input channels and output channels are divided into.
    /// * `kernel_shape`(`Option<Span<usize>>`) - The shape of the convolution kernel. If not present, should be inferred from input W.
    /// * `output_padding`(`Option<Span<usize>>`) - Additional elements added to the side with higher coordinate indices in the output. Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension. By default, this attribute is a zero vector. 
    /// * `output_shape`(`Option<Span<usize>>`) - The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified pads values are ignored. See doc for details for equations to generate pads.
    /// * `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
    /// * `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` that contains the result of the convolution transpose.
    ///
    /// ## Examples
    ///     
    /// ```rust
    /// use orion::operators::nn::NNTrait;
    /// use orion::numbers::FixedTrait;
    /// use orion::operators::nn::FP16x16NN;
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
    /// 
    /// fn example_conv_transpose() -> Tensor<FP16x16> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(2);
    ///     shape.append(3);
    ///     shape.append(3);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     let W = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(1);
    ///     shape.append(3);
    ///     shape.append(3);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 0, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 131072, sign: false });
    ///     data.append(FP16x16 { mag: 196608, sign: false });
    ///     data.append(FP16x16 { mag: 262144, sign: false });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     data.append(FP16x16 { mag: 393216, sign: false });
    ///     data.append(FP16x16 { mag: 458752, sign: false });
    ///     data.append(FP16x16 { mag: 524288, sign: false });
    ///     let mut X = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     return NNTrait::conv_transpose(
    ///         @X,
    ///         @W,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///         Option::None,
    ///     );
    /// 
    /// }
    /// >>> [
    ///         [
    ///             [
    ///                 [0.0, 1.0, 3.0, 3.0, 2.0],  
    ///                 [3.0, 8.0, 15.0, 12.0, 7.0],
    ///                 [9.0, 21.0, 36.0, 27.0, 15.0],
    ///                 [9.0, 20.0, 33.0, 24.0, 13.0],
    ///                 [6.0, 13.0, 21.0, 15.0, 8.0],
    ///             ],
    ///             [
    ///                 [0.0, 1.0, 3.0, 3.0, 2.0],
    ///                 [3.0, 8.0, 15.0, 12.0, 7.0],
    ///                 [9.0, 21.0, 36.0, 27.0, 15.0],
    ///                 [9.0, 20.0, 33.0, 24.0, 13.0],
    ///                 [6.0, 13.0, 21.0, 15.0, 8.0],
    ///             ],
    ///         ]
    ///     ]
    ///
    /// ````
    ///
    fn conv_transpose(
        X: @Tensor<T>,
        W: @Tensor<T>,
        B: Option<@Tensor<T>>,
        auto_pad: Option<orion::operators::nn::functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<T>;
    /// # NNTrait::col2im
    /// 
    /// ```rust
    ///     col2im(
    ///     data: @Tensor<T>,
    ///     image_shape: Span<usize>,
    ///     block_shape: Span<usize>,
    ///     dilations: Option<Span<usize>>,
    ///     pads: Option<Span<usize>>,
    ///     strides: Option<Span<usize>>,
    /// )  -> Tensor<T>
    /// ```
    /// 
    /// The operator rearranges column blocks back into a multidimensional image
    ///
    /// Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html, but it only supports batched multi-dimensional image tensors. Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.
    ///
    /// ## Args
    ///
    /// * `data`(`@Tensor<T>`) - Input data tensor to be rearranged from column blocks back into an image. This is a 3-dimensional tensor containing [N, C * n-ary-product(block_shape), L], where N is batch dimension, C is image channel dimension and L is number of blocks.
    /// * `image_shape`(`Span<usize>`) - The shape of the spatial dimensions of the image after rearranging the column blocks.This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img] for a 2-D image or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.
    /// * `block_shape`(`Span<usize>`) - The shape of the block to apply on the input.This is a 1-dimensional tensor of size of at least 2, containing the value [H_block, W_block] for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block.This is the block-shape before dilation is applied to it.
    /// * `dilations`(`Option<Span<usize>>`) - 1-dimensional tensor with dilation value along each spatial axis of the image. If not present, the dilation defaults to 1 along each spatial axis of the image.
    /// * `pads`(`Option<Span<usize>>`) - 1-dimensional tensor with padding value for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number of pixels added at the beginning of axis `i` and xi_end is the number of pixels added at the end of axis `i`. If not present, the padding defaults to 0 along start and end of each spatial axis.
    /// * `strides`(`Option<Span<usize>>`) - 1-dimensional tensor with stride value along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` output tensor produced by rearranging blocks into an image.
    ///
    /// ## Examples
    ///     
    /// ```rust
    /// use orion::operators::nn::NNTrait;
    /// use orion::numbers::FixedTrait;
    /// use orion::operators::nn::FP16x16NN;
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
    /// 
    /// fn example_col2im() -> Tensor<FP16x16> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(5);
    ///     shape.append(5);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 393216, sign: false });
    ///     data.append(FP16x16 { mag: 720896, sign: false });
    ///     data.append(FP16x16 { mag: 1048576, sign: false });
    ///     data.append(FP16x16 { mag: 1376256, sign: false });
    ///     data.append(FP16x16 { mag: 131072, sign: false });
    ///     data.append(FP16x16 { mag: 458752, sign: false });
    ///     data.append(FP16x16 { mag: 786432, sign: false });
    ///     data.append(FP16x16 { mag: 1114112, sign: false });
    ///     data.append(FP16x16 { mag: 1441792, sign: false });
    ///     data.append(FP16x16 { mag: 196608, sign: false });
    ///     data.append(FP16x16 { mag: 524288, sign: false });
    ///     data.append(FP16x16 { mag: 851968, sign: false });
    ///     data.append(FP16x16 { mag: 1179648, sign: false });
    ///     data.append(FP16x16 { mag: 1507328, sign: false });
    ///     data.append(FP16x16 { mag: 262144, sign: false });
    ///     data.append(FP16x16 { mag: 589824, sign: false });
    ///     data.append(FP16x16 { mag: 917504, sign: false });
    ///     data.append(FP16x16 { mag: 1245184, sign: false });
    ///     data.append(FP16x16 { mag: 1572864, sign: false });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     data.append(FP16x16 { mag: 0, sign: false });
    ///     data.append(FP16x16 { mag: 983040, sign: false });
    ///     data.append(FP16x16 { mag: 1310720, sign: false });
    ///     data.append(FP16x16 { mag: 1638400, sign: false });
    ///     let mut X = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let image_shape = array![5, 5].span();
    ///     let block_shape = array![1, 5].span();
    /// 
    ///     return NNTrait::col2im(
    ///         @X, image_shape, block_shape, Option::None, Option::None, Option::None,
    ///     );
    /// 
    /// 
    /// }
    /// >>> [
    ///         [
    ///             [
    ///                 [1.0, 2.0, 3.0, 4.0, 5.0],  
    ///                 [6.0, 7.0, 8.0, 9.0, 0.0],
    ///                 [11.0, 12.0, 13.0, 14.0, 15.0],
    ///                 [16.0, 17.0, 18.0, 19.0, 20.0],
    ///                 [21.0, 22.0, 23.0, 24.0, 25.0],
    ///             ]
    ///         ]
    ///     ]
    ///
    /// ````
    ///
    ///
    fn col2im(
        data: @Tensor<T>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<T>;
    /// # NNTrait::grid_sample
    /// 
    /// ```rust
    ///     fn grid_sample(
    ///     X: @Tensor<T>,
    ///     grid: @Tensor<T>,
    ///     align_corner: Option<usize>,
    ///     mode: Option<MODE>,
    ///     padding_mode: Option<PADDING_MODE>,
    /// ) -> Tensor<T>;
    /// ```
    /// 
    /// Given an input X and a flow-field grid, computes the output Y using X values and pixel locations from the grid.
    /// 
    /// ## Args
    ///
    /// * `X`(`@Tensor<T>`) - Input tensor of shape (N, C, D1, D2, ..., Dr), where N is the batch size, C is the number of channels, D1, D2, ..., Dr are the spatial dimensions.
    /// * `grid`(`@Tensor<T>`) - Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out, D2_out, ..., Dr_out are the spatial dimensions of the grid and output, and r is the number of spatial dimensions. Grid specifies the sampling locations normalized by the input spatial dimensions. 
    /// * `align_corners`(`Option<usize>`) - default is 0. If align_corners=1, the extrema are considered as referring to the center points of the input's corner pixels. If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels 
    /// * `mode`(`Option<MODE>`) - default is linear. Three interpolation modes: linear (default), nearest and cubic.
    /// * `padding_mode`(`Option<PADDING_MODE>`) - default is zeros. Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`.
    ///
    /// ## Returns
    ///
    /// A `Tensor<T>` of shape (N, C, D1_out, D2_out, ..., Dr_out) of the sampled values.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use orion::operators::nn::NNTrait;
    /// use orion::numbers::FixedTrait;
    /// use orion::operators::nn::FP16x16NN;
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
    ///
    /// fn example_grid_sample() -> Tensor<FP16x16> {
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(2);
    ///     shape.append(4);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 655360, sign: true });
    ///     data.append(FP16x16 { mag: 655360, sign: true });
    ///     data.append(FP16x16 { mag: 327680, sign: true });
    ///     data.append(FP16x16 { mag: 327680, sign: true });
    ///     data.append(FP16x16 { mag: 13107, sign: true });
    ///     data.append(FP16x16 { mag: 13107, sign: true });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 13107, sign: true });
    ///     data.append(FP16x16 { mag: 13107, sign: true });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    ///     data.append(FP16x16 { mag: 655360, sign: false });
    /// 
    ///     let mut grid = TensorTrait::new(shape.span(), data.span());
    /// 
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///     shape.append(1);
    ///     shape.append(3);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 0, sign: false });
    ///     data.append(FP16x16 { mag: 65536, sign: false });
    ///     data.append(FP16x16 { mag: 131072, sign: false });
    ///     data.append(FP16x16 { mag: 196608, sign: false });
    ///     data.append(FP16x16 { mag: 262144, sign: false });
    ///     data.append(FP16x16 { mag: 327680, sign: false });
    ///     let mut X = TensorTrait::new(shape.span(), data.span());
    /// 
    /// 
    ///     return NNTrait::grid_sample(
    ///         @X, @grid, Option::None, Option::None, Option::None,
    ///     );
    /// 
    /// }
    /// 
    ///}
    /// >>> [
    ///         [
    ///             [
    ///                 [0.0000, 0.0000, 1.7000, 0.0000], 
    ///                 [0.0000, 1.7000, 0.0000, 0.0000]
    ///             ]
    ///         ]
    ///     ]
    ///
    /// ````
    fn grid_sample(
        X: @Tensor<T>,
        grid: @Tensor<T>,
        align_corner: Option<usize>,
        mode: Option<orion::operators::nn::functional::grid_sample::MODE>,
        padding_mode: Option<orion::operators::nn::functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<T>;
}
