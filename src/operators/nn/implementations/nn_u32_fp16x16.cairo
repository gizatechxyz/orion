use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::{Tensor_u32_fp16x16, u32TensorAdd};
use orion::operators::tensor::implementations::tensor_fp16x16::{Tensor_fp16x16, FP16x16TensorDiv};

impl NN_u32_fp16x16 of NNTrait<u32, FP16x16> {
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<u32>) -> Tensor<FP16x16> {
        functional::sigmoid::sigmoid_from_int(*tensor)
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP16x16> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<FP16x16> {
        functional::softsign::softsign_from_int(*tensor)
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<FP16x16> {
        functional::softplus::softplus_from_int(*tensor)
    }

    fn linear(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<u32>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::leaky_relu::leaky_relu_from_int(*inputs, alpha)
    }
}
