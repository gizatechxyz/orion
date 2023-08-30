use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_i32_fp8x23::Tensor_i32_fp8x23;

impl NN_i32_fp8x23 of NNTrait<i32, FP8x23> {
    fn relu(tensor: @Tensor<i32>) -> Tensor<i32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<i32>) -> Tensor<FP8x23> {
        // sigmoid_i32(tensor).unwrap()
        panic(array![])
    }

    fn softmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FP8x23> {
        //  softmax_i32(tensor, axis)
        panic(array![])
    }

    fn logsoftmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<FP8x23> {
        //  logsoftmax_i32(tensor, axis)
        panic(array![])
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<FP8x23> {
        // softsign_i32(tensor).unwrap()
        panic(array![])
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<FP8x23> {
        // softplus_i32(tensor).unwrap()
        panic(array![])
    }

    fn linear(inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>) -> Tensor<i32> {
        // linear_i32(inputs, weights, bias)
        panic(array![])
    }

    fn leaky_relu(inputs: @Tensor<i32>, alpha: @FP8x23) -> Tensor<FP8x23> {
        // leaky_relu_i32(inputs, alpha).unwrap()
        panic(array![])
    }
}
