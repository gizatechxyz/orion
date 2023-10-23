use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::{
    FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorAdd
};

impl FP16x16NN of NNTrait<FP16x16> {
    fn relu(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP16x16>, weights: Tensor<FP16x16>, bias: Tensor<FP16x16>
    ) -> Tensor<FP16x16> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP16x16>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn hard_sigmoid(tensor: @Tensor<FP16x16>, alpha: @FP16x16, beta: @FP16x16) -> Tensor<FP16x16> {
        functional::hard_sigmoid::hard_sigmoid(*tensor, alpha, beta)
    }
}
