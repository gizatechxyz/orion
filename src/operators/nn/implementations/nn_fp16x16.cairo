use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::{
    FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorAdd
};
use orion::numbers::fixed_point::implementations::fp16x16wide::core::{
    FP16x16WImpl, FP16x16WTryIntoFP16x16, FP16x16W, FP16x16IntoFP16x16W
};
use orion::operators::tensor::implementations::tensor_fp16x16wide::{
    FP16x16WTensor, FP16x16WTensorDiv, FP16x16WTensorAdd
};

impl FP16x16NN of NNTrait<FP16x16> {
    fn relu(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax::softmaxWide::<FP16x16, u32, FP16x16W, u64>(tensor, axis)
    }

    fn softmax_zero(tensor: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        functional::softmax_zero::softmaxWide_zero::<FP16x16, u32, FP16x16W, u64>(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP16x16>, axis: usize) -> Tensor<FP16x16> {
        functional::logsoftmax::logsoftmaxWide::<FP16x16, u32, FP16x16W, u64>(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP16x16>, weights: Tensor<FP16x16>, bias: Tensor<FP16x16>
    ) -> Tensor<FP16x16> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP16x16>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP16x16>, alpha: @FP16x16) -> Tensor<FP16x16> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }

    fn hard_sigmoid(tensor: @Tensor<FP16x16>, alpha: @FP16x16, beta: @FP16x16) -> Tensor<FP16x16> {
        functional::hard_sigmoid::hard_sigmoid(*tensor, alpha, beta)
    }

    fn gemm(
        A: Tensor<FP16x16>,
        B: Tensor<FP16x16>,
        C: Option<Tensor<FP16x16>>,
        alpha: Option<FP16x16>,
        beta: Option<FP16x16>,
        transA: bool,
        transB: bool
    ) -> Tensor<FP16x16> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }
}
