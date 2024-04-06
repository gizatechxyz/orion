use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp32x32::core::{FP32x32, FP32x32Impl};
use orion::operators::tensor::implementations::tensor_fp32x32::{
    FP32x32Tensor, FP32x32TensorDiv, FP32x32TensorAdd, FP32x32TensorMul
};

impl FP32x32NN of NNTrait<FP32x32> {
    fn relu(tensor: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP32x32>, axis: Option<i32>) -> Tensor<FP32x32> {
        functional::softmax::softmax(tensor, axis)
    }

    fn softmax_zero(tensor: @Tensor<FP32x32>, axis: usize) -> Tensor<FP32x32> {
        functional::softmax_zero::softmax_zero(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP32x32>, axis: usize) -> Tensor<FP32x32> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP32x32>, weights: Tensor<FP32x32>, bias: Tensor<FP32x32>
    ) -> Tensor<FP32x32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP32x32>, alpha: @FP32x32) -> Tensor<FP32x32> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP32x32>, alpha: @FP32x32) -> Tensor<FP32x32> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }

    fn hard_sigmoid(tensor: @Tensor<FP32x32>, alpha: @FP32x32, beta: @FP32x32) -> Tensor<FP32x32> {
        functional::hard_sigmoid::hard_sigmoid(*tensor, alpha, beta)
    }

    fn hard_swish(tensor: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        functional::hard_swish::hard_swish(*tensor)
    }

    fn depth_to_space(
        tensor: @Tensor<FP32x32>, blocksize: usize, mode: felt252
    ) -> Tensor<FP32x32> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<FP32x32>, blocksize: usize) -> Tensor<FP32x32> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
    }

    fn gemm(
        A: Tensor<FP32x32>,
        B: Tensor<FP32x32>,
        C: Option<Tensor<FP32x32>>,
        alpha: Option<FP32x32>,
        beta: Option<FP32x32>,
        transA: bool,
        transB: bool
    ) -> Tensor<FP32x32> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn grid_sample(
        X: @Tensor<FP32x32>,
        grid: @Tensor<FP32x32>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<FP32x32> {
        functional::grid_sample::grid_sample(X, grid, align_corner, mode, padding_mode)
    }

    fn col2im(
        data: @Tensor<FP32x32>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP32x32> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<FP32x32>,
        W: @Tensor<FP32x32>,
        B: Option<@Tensor<FP32x32>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP32x32> {
        functional::conv_transpose::conv_transpose(
            X,
            W,
            B,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            output_padding,
            output_shape,
            pads,
            strides
        )
    }

    fn conv(
        X: @Tensor<FP32x32>,
        W: @Tensor<FP32x32>,
        B: Option<Span<FP32x32>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP32x32> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }
}
