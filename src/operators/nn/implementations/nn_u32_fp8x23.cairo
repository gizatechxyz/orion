use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_u32_fp8x23::{Tensor_u32_fp8x23, u32TensorAdd};
use orion::operators::tensor::implementations::tensor_fp8x23::{Tensor_fp8x23, FP8x23TensorDiv};

impl NN_u32_fp8x23 of NNTrait<u32, FP8x23> {
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        functional::sigmoid::sigmoid_from_int(*tensor)
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP8x23> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP8x23> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        functional::softsign::softsign_from_int(*tensor)
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        functional::softplus::softplus_from_int(*tensor)
    }

    fn linear(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<u32>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::leaky_relu::leaky_relu_from_int(*inputs, alpha)
    }
}
