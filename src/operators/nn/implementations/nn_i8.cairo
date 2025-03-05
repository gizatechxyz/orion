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

    fn softmax(tensor: @Tensor<i8>, axis: Option<i32>) -> Tensor<i8> {
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

    fn depth_to_space(tensor: @Tensor<i8>, blocksize: usize, mode: felt252) -> Tensor<i8> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<i8>, blocksize: usize) -> Tensor<i8> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
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

    fn grid_sample(
        X: @Tensor<i8>,
        grid: @Tensor<i8>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<i8> {
        panic(array!['not supported!'])
    }

    fn col2im(
        data: @Tensor<i8>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i8> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<i8>,
        W: @Tensor<i8>,
        B: Option<@Tensor<i8>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i8> {
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

    fn roi_align(
        X: @Tensor<i8>,
        roi: @Tensor<i8>,
        batch_indices: @Tensor<usize>,
        coordinate_transformation_mode: Option<
            orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE
        >,
        mode: Option<orion::operators::nn::functional::roi_align::MODE>,
        output_height: Option<usize>,
        output_width: Option<usize>,
        sampling_ratio: Option<i8>,
        spatial_scale: Option<i8>,
    ) -> Tensor<i8> {
        panic(array!['not supported!'])
    }
}
