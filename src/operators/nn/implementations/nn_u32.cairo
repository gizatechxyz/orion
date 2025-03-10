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

    fn softmax(tensor: @Tensor<u32>, axis: Option<i32>) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn softmax_zero(tensor: @Tensor<u32>, axis: usize) -> Tensor<u32> {
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

    fn hard_sigmoid(tensor: @Tensor<u32>, alpha: @u32, beta: @u32) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn hard_swish(tensor: @Tensor<u32>) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn depth_to_space(tensor: @Tensor<u32>, blocksize: usize, mode: felt252) -> Tensor<u32> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<u32>, blocksize: usize) -> Tensor<u32> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
    }

    fn gemm(
        A: Tensor<u32>,
        B: Tensor<u32>,
        C: Option<Tensor<u32>>,
        alpha: Option<u32>,
        beta: Option<u32>,
        transA: bool,
        transB: bool
    ) -> Tensor<u32> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn grid_sample(
        X: @Tensor<u32>,
        grid: @Tensor<u32>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<u32> {
        panic(array!['not supported!'])
    }

    fn col2im(
        data: @Tensor<u32>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<u32> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<u32>,
        W: @Tensor<u32>,
        B: Option<@Tensor<u32>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<u32> {
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
        X: @Tensor<u32>,
        W: @Tensor<u32>,
        B: Option<Span<u32>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<u32> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }
}
