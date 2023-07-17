use core::array::ArrayTrait;
use core::option::OptionTrait;
use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::functional::relu::relu_u32::relu_u32;
use orion::operators::nn::functional::sigmoid::sigmoid_u32::core::sigmoid_u32;
use orion::operators::nn::functional::softmax::softmax_u32::softmax_u32;
use orion::operators::nn::functional::logsoftmax::logsoftmax_u32::logsoftmax_u32;
use orion::operators::nn::functional::softsign::softsign_u32::core::softsign_u32;
use orion::operators::nn::functional::softplus::softplus_u32::core::softplus_u32;
use orion::operators::nn::functional::linear::linear_u32::linear_u32;
use orion::operators::nn::functional::leaky_relu::leaky_relu_u32::core::leaky_relu_u32;
use orion::numbers::fixed_point::core::{FixedType};



impl NN_u32 of NNTrait<u32> {
    fn relu(tensor: @Tensor<u32>, threshold: u32) -> Tensor<u32> {
        relu_u32(tensor, threshold)
    }

    fn sigmoid(tensor: @Tensor<u32>) -> Tensor<FixedType> {
        sigmoid_u32(tensor).unwrap()
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FixedType> {
        softmax_u32(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FixedType> {
        logsoftmax_u32(tensor, axis)
    }

    fn softsign(tensor: @Tensor<u32>) -> Tensor<FixedType> {
        softsign_u32(tensor).unwrap()
    }

    fn softplus(tensor: @Tensor<u32>) -> Tensor<FixedType> {
        softplus_u32(tensor).unwrap()
    }

    fn linear(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>) -> Tensor<u32> {
        linear_u32(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<u32>, alpha: @FixedType, threshold: u32) -> Tensor<FixedType> {
        leaky_relu_u32(inputs, alpha, threshold).unwrap()
    }

    fn convint(inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>, kernel_size: usize, strides: usize) -> Tensor<u32> {
        let mut data = ArrayTrait::new();
        data.append('Not Implemented');
        return panic(data);
    } 
}
