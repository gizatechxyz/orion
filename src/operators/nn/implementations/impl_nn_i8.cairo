use core::option::OptionTrait;
use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_i8::relu_i8;
use orion::operators::nn::functional::sigmoid::sigmoid_i8::core::sigmoid_i8;
use orion::operators::nn::functional::softmax::softmax_i8::softmax_i8;
use orion::operators::nn::functional::logsoftmax::logsoftmax_i8::logsoftmax_i8;
use orion::operators::nn::functional::softsign::softsign_i8::core::softsign_i8;
use orion::operators::nn::functional::softplus::softplus_i8::core::softplus_i8;
use orion::operators::nn::functional::linear::linear_i8::linear_i8;
use orion::operators::nn::functional::leaky_relu::leaky_relu_i8::core::leaky_relu_i8;
use orion::numbers::fixed_point::core::{FixedType};


impl NN_i8 of NNTrait<i8> {
    fn relu(tensor: @Tensor<i8>, threshold: i8) -> Tensor<i8> {
        relu_i8(tensor, threshold)
    }

    fn sigmoid(tensor: @Tensor<i8>) -> Tensor<FixedType> {
        sigmoid_i8(tensor).unwrap()
    }

    fn softmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FixedType> {
        softmax_i8(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FixedType> {
        logsoftmax_i8(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i8>) -> Tensor<FixedType> {
        softsign_i8(tensor).unwrap()
    }

    fn softplus(tensor: @Tensor<i8>) -> Tensor<FixedType> {
        softplus_i8(tensor).unwrap()
    }

    fn linear(inputs: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>) -> Tensor<i8> {
        linear_i8(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i8>, alpha: @FixedType) -> Tensor<FixedType> {
        leaky_relu_i8(inputs, alpha).unwrap()
    }
}
