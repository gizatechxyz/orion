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

    fn softmax(tensor: @Tensor<FP16x16>, axis: Option<i32>) -> Tensor<FP16x16> {
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

    fn depth_to_space(
        tensor: @Tensor<FP16x16>, blocksize: usize, mode: felt252
    ) -> Tensor<FP16x16> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<FP16x16>, blocksize: usize) -> Tensor<FP16x16> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
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

    fn grid_sample(
        X: @Tensor<FP16x16>,
        grid: @Tensor<FP16x16>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<FP16x16> {
        functional::grid_sample::grid_sample(X, grid, align_corner, mode, padding_mode)
    }

    fn col2im(
        data: @Tensor<FP16x16>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP16x16> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<FP16x16>,
        W: @Tensor<FP16x16>,
        B: Option<@Tensor<FP16x16>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP16x16> {
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
        X: @Tensor<FP16x16>,
        W: @Tensor<FP16x16>,
        B: Option<Span<FP16x16>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP16x16> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }

    fn roi_align(
        X: @Tensor<FP16x16>,
        roi: @Tensor<FP16x16>,
        batch_indices: @Tensor<usize>,
        coordinate_transformation_mode: Option<
            orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE
        >,
        mode: Option<orion::operators::nn::functional::roi_align::MODE>,
        output_height: Option<usize>,
        output_width: Option<usize>,
        sampling_ratio: Option<FP16x16>,
        spatial_scale: Option<FP16x16>,
    ) -> Tensor<FP16x16> {
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
