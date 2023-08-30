use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;

impl NN_fp8x23 of NNTrait<FP8x23, FP8x23> {
    fn relu(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        //sigmoid_fp(tensor).unwrap()
        panic(array![])
    }

    fn softmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        //softmax_fp(tensor, axis)
        panic(array![])
    }

    fn logsoftmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        //logsoftmax_fp(tensor, axis)
        panic(array![])
    }

    fn softsign(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        // softsign_fp(tensor).unwrap()
        panic(array![])
    }

    fn softplus(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        //  softplus_fp(tensor).unwrap()
        panic(array![])
    }

    fn linear(
        inputs: Tensor<FP8x23>, weights: Tensor<FP8x23>, bias: Tensor<FP8x23>
    ) -> Tensor<FP8x23> {
        // linear_fp(inputs, weights, bias)
        panic(array![])
    }

    fn leaky_relu(inputs: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        // leaky_relu_fp(inputs, alpha).unwrap()
        panic(array![])
    }
}
