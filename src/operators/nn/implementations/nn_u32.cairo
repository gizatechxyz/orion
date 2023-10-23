use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::operators::tensor::implementations::tensor_u32::{U32Tensor, U32TensorAdd};

impl U32NN of NNTrait<u32> {
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<u32>) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn logsoftmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn linear(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<u32>, alpha: @u32) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn thresholded_relu(tensor: @Tensor<u32>, alpha: @u32) -> Tensor<u32> {
        panic(array!['not supported!'])
    }
}
