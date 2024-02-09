use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::operators::tensor::implementations::tensor_i8::{I8Tensor, I8TensorAdd};

impl I8NN of NNTrait<i8> {
    fn relu(tensor: @Tensor<i8>) -> Tensor<i8> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn softmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn softmax_zero(tensor: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn logsoftmax(tensor: @Tensor<i8>, axis: usize) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn softsign(tensor: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn softplus(tensor: @Tensor<i8>) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn linear(inputs: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>) -> Tensor<i8> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i8>, alpha: @i8) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn thresholded_relu(tensor: @Tensor<i8>, alpha: @i8) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn hard_sigmoid(tensor: @Tensor<i8>, alpha: @i8, beta: @i8) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn gemm(
        A: Tensor<i8>,
        B: Tensor<i8>,
        C: Option<Tensor<i8>>,
        alpha: Option<i8>,
        beta: Option<i8>,
        transA: bool,
        transB: bool
    ) -> Tensor<i8> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn conv(
        X: @Tensor<i8>,
        W: @Tensor<i8>,
        B: Option<Span<i8>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i8> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }
}
