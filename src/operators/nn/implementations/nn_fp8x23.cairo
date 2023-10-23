use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::{
    FP8x23Tensor, FP8x23TensorDiv, FP8x23TensorAdd
};

impl FP8x23NN of NNTrait<FP8x23> {
    fn relu(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP8x23>, weights: Tensor<FP8x23>, bias: Tensor<FP8x23>
    ) -> Tensor<FP8x23> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }
}
