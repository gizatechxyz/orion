use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_i8_fp16x16::{Tensor_i8_fp16x16, i8TensorAdd};
use orion::operators::tensor::implementations::tensor_fp16x16::{Tensor_fp16x16, FP16x16TensorDiv};

impl NN_i8_fp16x16 of NNTrait<i8, FP16x16> {
    fn relu(tensor: @Tensor<i8>) -> Tensor<i8> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<i8>) -> Tensor<FP16x16> {
        functional::sigmoid::sigmoid_from_int(*tensor)
    }

    fn softmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FP16x16> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i8>) -> Tensor<FP16x16> {
        functional::softsign::softsign_from_int(*tensor)
    }

    fn softplus(tensor: @Tensor<i8>) -> Tensor<FP16x16> {
        functional::softplus::softplus_from_int(*tensor)
    }

    fn linear(inputs: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>) -> Tensor<i8> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i8>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::leaky_relu::leaky_relu_from_int(*inputs, alpha)
    }
}
