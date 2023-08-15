use core::option::OptionTrait;
use orion::operators::tensor::core::Tensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional::relu::relu_i32::relu_i32;
use orion::operators::nn::functional::sigmoid::sigmoid_i32::core::sigmoid_i32;
use orion::operators::nn::functional::softmax::softmax_i32::softmax_i32;
use orion::operators::nn::functional::logsoftmax::logsoftmax_fp::logsoftmax_fp;
use orion::operators::nn::functional::softsign::softsign_i32::core::softsign_i32;
use orion::operators::nn::functional::softplus::softplus_i32::core::softplus_i32;
use orion::operators::nn::functional::linear::linear_fp::linear_fp;
use orion::operators::nn::functional::leaky_relu::leaky_relu_fp::core::leaky_relu_fp;
use orion::numbers::fixed_point::core::{FixedType};


impl NN_fp of NNTrait<FixedType> {
    fn relu(tensor: @Tensor<FixedType>) -> Tensor<FixedType> {
        panic(array![''])
    }

    fn sigmoid(tensor: @Tensor<FixedType>) -> Tensor<FixedType> {
        panic(array![''])
    }

    fn softmax(tensor: @Tensor<FixedType>, axis: usize) -> Tensor<FixedType> {
        panic(array![''])
    }

    fn logsoftmax(tensor: @Tensor<FixedType>, axis: usize) -> Tensor<FixedType> {
        logsoftmax_fp(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FixedType>) -> Tensor<FixedType> {
        panic(array![''])
    }

    fn softplus(tensor: @Tensor<FixedType>) -> Tensor<FixedType> {
        panic(array![''])
    }

    fn linear(
        inputs: Tensor<FixedType>, weights: Tensor<FixedType>, bias: Tensor<FixedType>
    ) -> Tensor<FixedType> {
        linear_fp(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FixedType>, alpha: @FixedType) -> Tensor<FixedType> {
        leaky_relu_fp(inputs, alpha).unwrap()
    }
}
