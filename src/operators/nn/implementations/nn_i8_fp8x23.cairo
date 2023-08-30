use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::nn::core::{NNTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;

impl NN_i8_fp8x23 of NNTrait<i8, FP8x23> {
    fn relu(tensor: @Tensor<i8>) -> Tensor<i8> {
        //relu_i8(*tensor)
        panic(array![])
    }

    fn sigmoid(tensor: @Tensor<i8>) -> Tensor<FP8x23> {
        // sigmoid_i8(tensor).unwrap()
        panic(array![])
    }

    fn softmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FP8x23> {
        //  softmax_i8(tensor, axis)
        panic(array![])
    }

    fn logsoftmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<FP8x23> {
        //  logsoftmax_i8(tensor, axis)
        panic(array![])
    }

    fn softsign(tensor: @Tensor<i8>) -> Tensor<FP8x23> {
        // softsign_i8(tensor).unwrap()
        panic(array![])
    }

    fn softplus(tensor: @Tensor<i8>) -> Tensor<FP8x23> {
        // softplus_i8(tensor).unwrap()
        panic(array![])
    }

    fn linear(inputs: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>) -> Tensor<i8> {
        // linear_i8(inputs, weights, bias)
        panic(array![])
    }

    fn leaky_relu(inputs: @Tensor<i8>, alpha: @FP8x23) -> Tensor<FP8x23> {
        // leaky_relu_i8(inputs, alpha).unwrap()
        panic(array![])
    }
}
