use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::functional::relu::relu_u32::relu_u32;
use onnx_cairo::operators::nn::functional::softmax::softmax_u32::softmax_u32;
use onnx_cairo::operators::nn::functional::linear::linear_u32::linear_u32;
use onnx_cairo::numbers::fixed_point::core::FixedType;

impl u32NN of NNTrait<u32> {
    fn relu(tensor: @Tensor<u32>) -> Tensor<u32> {
        relu_u32(tensor)
    }

    fn softmax(tensor: @Tensor<u32>, axis: usize) -> Tensor<FixedType> {
        softmax_u32(tensor, axis)
    }

    fn linear(
        inputs: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>, quantized: bool
    ) -> Tensor<u32> {
        linear_u32(inputs, weights, bias, quantized)
    }
}
