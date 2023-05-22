use onnx_cairo::numbers::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::performance::core::PerfomanceTrait;
use onnx_cairo::performance::functional::quantization::quant_i32::quantize_tensor;

impl i32Performance of PerfomanceTrait<i32> {
    fn quantize_linear(self: @Tensor<i32>) -> Tensor<i32> {
        quantize_tensor(self)
    }
}
