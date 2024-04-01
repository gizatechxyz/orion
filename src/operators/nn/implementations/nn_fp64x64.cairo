use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp64x64::core::{FP64x64, FP64x64Impl};
use orion::operators::tensor::implementations::tensor_fp64x64::{
    FP64x64Tensor, FP64x64TensorDiv, FP64x64TensorAdd
};

impl FP64x64NN of NNTrait<FP64x64> {
    fn relu(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP64x64>, axis: Option<i32>) -> Tensor<FP64x64> {
        functional::softmax::softmax(tensor, axis)
    }

    fn softmax_zero(tensor: @Tensor<FP64x64>, axis: usize) -> Tensor<FP64x64> {
        functional::softmax_zero::softmax_zero(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP64x64>, axis: usize) -> Tensor<FP64x64> {
        functional::logsoftmax::logsoftmax(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP64x64>) -> Tensor<FP64x64> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP64x64>, weights: Tensor<FP64x64>, bias: Tensor<FP64x64>
    ) -> Tensor<FP64x64> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP64x64>, alpha: @FP64x64) -> Tensor<FP64x64> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP64x64>, alpha: @FP64x64) -> Tensor<FP64x64> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }

    fn hard_sigmoid(tensor: @Tensor<FP64x64>, alpha: @FP64x64, beta: @FP64x64) -> Tensor<FP64x64> {
        functional::hard_sigmoid::hard_sigmoid(*tensor, alpha, beta)
    }

    fn depth_to_space(
        tensor: @Tensor<FP64x64>, blocksize: usize, mode: felt252
    ) -> Tensor<FP64x64> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<FP64x64>, blocksize: usize) -> Tensor<FP64x64> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
    }

    fn gemm(
        A: Tensor<FP64x64>,
        B: Tensor<FP64x64>,
        C: Option<Tensor<FP64x64>>,
        alpha: Option<FP64x64>,
        beta: Option<FP64x64>,
        transA: bool,
        transB: bool
    ) -> Tensor<FP64x64> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn grid_sample(
        X: @Tensor<FP64x64>,
        grid: @Tensor<FP64x64>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<FP64x64> {
        functional::grid_sample::grid_sample(X, grid, align_corner, mode, padding_mode)
    }

    fn col2im(
        data: @Tensor<FP64x64>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP64x64> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<FP64x64>,
        W: @Tensor<FP64x64>,
        B: Option<@Tensor<FP64x64>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP64x64> {
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
        X: @Tensor<FP64x64>,
        W: @Tensor<FP64x64>,
        B: Option<Span<FP64x64>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP64x64> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }

    fn roi_align(
        X: @Tensor<FP64x64>,
        roi: @Tensor<FP64x64>,
        batch_indices: @Tensor<usize>,
        coordinate_transformation_mode: Option<
            orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE
        >,
        mode: Option<orion::operators::nn::functional::roi_align::MODE>,
        output_height: Option<usize>,
        output_width: Option<usize>,
        sampling_ratio: Option<FP64x64>,
        spatial_scale: Option<FP64x64>,
    ) -> Tensor<FP64x64> {
        functional::roi_align::roi_align(
            X,
            roi,
            batch_indices,
            coordinate_transformation_mode,
            mode,
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
        )
    }
}
