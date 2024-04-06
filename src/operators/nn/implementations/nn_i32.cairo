use orion::operators::tensor::core::Tensor;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::operators::tensor::implementations::tensor_i32::{I32Tensor, I32TensorAdd};

impl I32NN of NNTrait<i32> {
    fn relu(tensor: @Tensor<i32>) -> Tensor<i32> {
        functional::relu::relu(*tensor)
    }

    fn sigmoid(tensor: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn softmax(tensor: @Tensor<i32>, axis: Option<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn softmax_zero(tensor: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn logsoftmax(tensor: @Tensor<i32>, axis: usize) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn softsign(tensor: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn softplus(tensor: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }
    

    fn linear(inputs: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>) -> Tensor<i32> {
        functional::linear::linear(inputs, weights, bias)
    }

    fn leaky_relu(inputs: @Tensor<i32>, alpha: @i32) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn thresholded_relu(tensor: @Tensor<i32>, alpha: @i32) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn hard_sigmoid(tensor: @Tensor<i32>, alpha: @i32, beta: @i32) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn hard_swish(tensor: @Tensor<i32>) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn depth_to_space(tensor: @Tensor<i32>, blocksize: usize, mode: felt252) -> Tensor<i32> {
        functional::depth_to_space::depth_to_space(*tensor, blocksize, mode)
    }

    fn space_to_depth(tensor: @Tensor<i32>, blocksize: usize) -> Tensor<i32> {
        functional::space_to_depth::space_to_depth(*tensor, blocksize)
    }

    fn gemm(
        A: Tensor<i32>,
        B: Tensor<i32>,
        C: Option<Tensor<i32>>,
        alpha: Option<i32>,
        beta: Option<i32>,
        transA: bool,
        transB: bool
    ) -> Tensor<i32> {
        functional::gemm::gemm(A, B, C, alpha, beta, transA, transB)
    }

    fn grid_sample(
        X: @Tensor<i32>,
        grid: @Tensor<i32>,
        align_corner: Option<usize>,
        mode: Option<functional::grid_sample::MODE>,
        padding_mode: Option<functional::grid_sample::PADDING_MODE>,
    ) -> Tensor<i32> {
        panic(array!['not supported!'])
    }

    fn col2im(
        data: @Tensor<i32>,
        image_shape: Span<usize>,
        block_shape: Span<usize>,
        dilations: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i32> {
        functional::col2im::col2im(data, image_shape, block_shape, dilations, pads, strides,)
    }

    fn conv_transpose(
        X: @Tensor<i32>,
        W: @Tensor<i32>,
        B: Option<@Tensor<i32>>,
        auto_pad: Option<functional::conv_transpose::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        output_padding: Option<Span<usize>>,
        output_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i32> {
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
        X: @Tensor<i32>,
        W: @Tensor<i32>,
        B: Option<Span<i32>>,
        auto_pad: Option<functional::conv::AUTO_PAD>,
        dilations: Option<Span<usize>>,
        group: Option<usize>,
        kernel_shape: Option<Span<usize>>,
        pads: Option<Span<usize>>,
        strides: Option<Span<usize>>,
    ) -> Tensor<i32> {
        functional::conv::conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)
    }
}
