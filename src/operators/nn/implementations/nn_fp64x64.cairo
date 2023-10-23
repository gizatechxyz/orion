use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp64x64::core::{FP64x64, FP64x64Impl};
use orion::operators::tensor::implementations::tensor_fp64x64::{
    FP64x64Tensor, FP64x64TensorDiv, FP64x64TensorAdd
};

impl FP64x64NN of NNTrait<FP64x64> {
    fn relu(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP64x64>, axis: usize) -> Tensor<FP64x64> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP64x64>, axis: usize) -> Tensor<FP64x64> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP64x64>, weights: Tensor<FP64x64>, bias: Tensor<FP64x64>
    ) -> Tensor<FP64x64> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP64x64>, alpha: @FP64x64) -> Tensor<FP64x64> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP64x64>, alpha: @FP64x64) -> Tensor<FP64x64> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }
}
