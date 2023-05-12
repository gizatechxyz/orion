use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::numbers::fixed_point::core::FixedType;

trait PerfomanceTrait<T> {
    fn quantize_linear(self: @Tensor<T>) -> Tensor<T>;
}
