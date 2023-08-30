use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_i32_fp16x16::{Tensor_i32_fp16x16, i32TensorAdd};
use orion::operators::tensor::implementations::tensor_fp16x16::{Tensor_fp16x16, FP16x16TensorDiv};

impl NN_i32_fp16x16 of NNTrait<i32, FP16x16> {
    fn relu(tensor: @Tensor<i32>) -> Tensor<i32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<i32>) -> Tensor<FP16x16> {
        functional::sigmoid::sigmoid_from_int(*tensor)
    }

    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FP16x16> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<FP16x16> {
        functional::softsign::softsign_from_int(*tensor)
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<FP16x16> {
        functional::softplus::softplus_from_int(*tensor)
    }

    fn linear(inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>) -> Tensor<i32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i32>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::leaky_relu::leaky_relu_from_int(*inputs, alpha)
    }
}
