use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_u32::{quantize_tensor, quantize_fp_tensor};

impl Performance_i32_fp of PerfomanceTrait<u32> {
    fn quantize_linear(self: @Tensor<u32>) -> Tensor<u32> {
        quantize_tensor(self)
    }

    fn quantize_linear_from_fp(self: @Tensor<FixedType>) -> Tensor<u32> {
        quantize_fp_tensor(self)
    }
}
