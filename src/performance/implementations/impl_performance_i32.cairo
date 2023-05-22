use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_i32::quantize_tensor;

impl i32Performance of PerfomanceTrait<i32> {
    fn quantize_linear(self: @Tensor<i32>) -> Tensor<i32> {
        quantize_tensor(self)
    }
}
