use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::{
    FP8x23Tensor, FP8x23TensorDiv, FP8x23TensorAdd
};
use orion::numbers::fixed_point::implementations::fp8x23wide::core::{
    FP8x23WImpl, FP8x23WTryIntoFP8x23, FP8x23W, FP8x23IntoFP8x23W
};
use orion::operators::tensor::implementations::tensor_fp8x23wide::{FP8x23WTensor};

impl FP8x23NN of NNTrait<FP8x23> {
    fn relu(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::sigmoid::sigmoid(*tensor)
    }

    fn softmax(tensor: @Tensor<FP8x23>, axis: Option<i32>) -> Tensor<FP8x23> {
        functional::softmax::softmaxWide::<FP8x23, u32, FP8x23W, u64>(tensor, axis)
    }

    fn softmax_zero(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        functional::softmax_zero::softmaxWide_zero::<FP8x23, u32, FP8x23W, u64>(tensor, axis)
    }

    fn logsoftmax(tensor: @Tensor<FP8x23>, axis: usize) -> Tensor<FP8x23> {
        functional::logsoftmax::logsoftmaxWide::<FP8x23, u32, FP8x23W, u64>(tensor, axis)
    }

    fn softsign(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softsign::softsign(*tensor)
    }

    fn softplus(tensor: @Tensor<FP8x23>) -> Tensor<FP8x23> {
        functional::softplus::softplus(*tensor)
    }

    fn linear(
        inputs: Tensor<FP8x23>, weights: Tensor<FP8x23>, bias: Tensor<FP8x23>
    ) -> Tensor<FP8x23> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::leaky_relu::leaky_relu(*inputs, alpha)
    }

    fn thresholded_relu(tensor: @Tensor<FP8x23>, alpha: @FP8x23) -> Tensor<FP8x23> {
        functional::thresholded_relu::thresholded_relu(*tensor, alpha)
    }

    fn hard_sigmoid(tensor: @Tensor<FP8x23>, alpha: @FP8x23, beta: @FP8x23) -> Tensor<FP8x23> {
        functional::hard_sigmoid::hard_sigmoid(*tensor, alpha, beta)
    }

    fn depth_to_space(tensor: @Tensor<FP8x23>, blocksize: usize, mode: felt252) -> Tensor<FP8x23> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<FP8x23>, blocksize: usize) -> Tensor<FP8x23> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
    }

    fn gemm(
        A: Tensor<FP8x23>,
        B: Tensor<FP8x23>,
        C: Option<Tensor<FP8x23>>,
        alpha: Option<FP8x23>,
        beta: Option<FP8x23>,
        transA: bool,
        transB: bool
    ) -> Tensor<FP8x23> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn grid_sample(
        X: @Tensor<FP8x23>,
        grid: @Tensor<FP8x23>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<FP8x23> {
        functional::grid_sample::grid_sample(X, grid, align_corner, mode, padding_mode)
    }

    fn col2im(
        data: @Tensor<FP8x23>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP8x23> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<FP8x23>,
        W: @Tensor<FP8x23>,
        B: Option<@Tensor<FP8x23>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP8x23> {
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
        X: @Tensor<FP8x23>,
        W: @Tensor<FP8x23>,
        B: Option<Span<FP8x23>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<FP8x23> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }

    fn roi_align(
        X: @Tensor<FP8x23>,
        roi: @Tensor<FP8x23>,
        batch_indices: @Tensor<usize>,
        coordinate_transformation_mode: Option<functional::roi_align::TRANSFORMATION_MODE>,
        mode: Option<functional::roi_align::MODE>,
        output_height: Option<usize>,
        output_width: Option<usize>,
        sampling_ratio: Option<FP8x23>,
        spatial_scale: Option<FP8x23>,
    ) -> Tensor<FP8x23> {
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
