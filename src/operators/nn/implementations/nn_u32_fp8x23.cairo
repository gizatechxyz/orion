use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::{NNTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;

impl NN_u32_fp8x23 of NNTrait<u32, FP8x23> {
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        //relu_u32(*tensor)
        panic(array![])
    }

    fn sigmoid(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        // sigmoid_u32(tensor).unwrap()
        panic(array![])
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP8x23> {
        //  softmax_u32(tensor, axis)
        panic(array![])
    }

    fn logsoftmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FP8x23> {
        //  logsoftmax_u32(tensor, axis)
        panic(array![])
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        // softsign_u32(tensor).unwrap()
        panic(array![])
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<FP8x23> {
        // softplus_u32(tensor).unwrap()
        panic(array![])
    }

    fn linear(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
        // linear_u32(inputs, weights, bias)
        panic(array![])
    }

    fn leaky_relu(inputs: @Tensor<u32>, alpha: @FP8x23) -> Tensor<FP8x23> {
        // leaky_relu_u32(inputs, alpha).unwrap()
        panic(array![])
    }
}
