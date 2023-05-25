use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional::relu::relu_i32::relu_i32;
use orion::operators::nn::functional::softmax::softmax_i32::softmax_i32;
use orion::operators::nn::functional::softsign::softsign_i32::softsign_i32;
use orion::operators::nn::functional::softplus::softplus_i32::softplus_i32;
use orion::operators::nn::functional::linear::linear_i32::linear_i32;
use orion::operators::nn::functional::leaky_relu::leaky_relu_i32::leaky_relu_i32;
use orion::numbers::fixed_point::core::FixedType;

impl i32NN of NNTrait<i32> {
    fn relu(tensor: @Tensor<i32>, threshold: i32) -> Tensor<i32> {
        relu_i32(tensor, threshold)
    }

    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
        softmax_i32(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<FixedType> {
        softsign_i32(tensor)
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<FixedType> {
        softplus_i32(tensor)
    }

    fn linear(
        inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>, quantized: bool
    ) -> Tensor<i32> {
        linear_i32(inputs, weights, bias, quantized)
    }

    fn leaky_relu(inputs: @Tensor<i32>, alpha: @FixedType, threshold: i32) -> Tensor<FixedType> {
        leaky_relu_i32(inputs, alpha, threshold)
    }
}
