use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::performance::core::PerfomanceTrait;
use onnx_cairo::performance::functional::quantization::quant_u32::quantize_tensor;

impl u32Performance of PerfomanceTrait<u32> {
    fn quantize_linear(self: @Tensor<u32>) -> Tensor<u32> {
        quantize_tensor(self)
    }
}
