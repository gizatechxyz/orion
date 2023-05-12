use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::numbers::fixed_point::core::FixedType;

trait NNTrait<T> {
    fn relu(tensor: @Tensor<T>) -> Tensor<T>;
    fn softmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>;
    fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>, quantized: bool) -> Tensor<T>;
}
