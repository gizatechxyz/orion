use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::{Tensor_fp8x23, FP8x23TensorDiv};

impl NN_fp8x23 of NNTrait<FP8x23, FP8x23> {
    fn relu(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::sigmoid::sigmoid_from_fp(*tensor)
    }

    fn softmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        functional::softmax::softmax(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        //logsoftmax_fp(tensor, axis)
        panic(array![])
    }

    fn softsign(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softsign::softsign_from_fp(*tensor)
    }

    fn softplus(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softplus::softplus_from_fp(*tensor)
    }

    fn linear(
        inputs: Tensor<FP8x23>, weights: Tensor<FP8x23>, bias: Tensor<FP8x23>
    ) -> Tensor<FP8x23> {
        // linear_fp(inputs, weights, bias)
        panic(array![])
    }

    fn leaky_relu(inputs: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::leaky_relu::leaky_relu_from_fp(*inputs, alpha)
    }
}
