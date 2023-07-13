use core::option::OptionTrait;
use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_i32::relu_i32;
use orion::operators::nn::functional::sigmoid::sigmoid_i32::core::sigmoid_i32;
use orion::operators::nn::functional::softmax::softmax_i32::softmax_i32;
use orion::operators::nn::functional::logsoftmax::logsoftmax_i32::logsoftmax_i32;
use orion::operators::nn::functional::softsign::softsign_i32::core::softsign_i32;
use orion::operators::nn::functional::softplus::softplus_i32::core::softplus_i32;
use orion::operators::nn::functional::linear::linear_i32::linear_i32;
use orion::operators::nn::functional::leaky_relu::leaky_relu_i32::core::leaky_relu_i32;
use orion::numbers::fixed_point::core::{FixedType};


impl NN_i32 of NNTrait<i32> {
    fn relu(tensor: @Tensor<i32>, threshold: i32) -> Tensor<i32> {
        relu_i32(tensor, threshold)
    }

    fn sigmoid(tensor: @Tensor<i32>) -> Tensor<FixedType> {
        sigmoid_i32(tensor).unwrap()
    }

    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
        softmax_i32(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
        logsoftmax_i32(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<FixedType> {
        softsign_i32(tensor).unwrap()
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<FixedType> {
        softplus_i32(tensor).unwrap()
    }

    fn linear(inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>) -> Tensor<i32> {
        linear_i32(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i32>, alpha: @FixedType, threshold: i32) -> Tensor<FixedType> {
        leaky_relu_i32(inputs, alpha, threshold).unwrap()
    }
}
