use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_u32::{quantize_tensor, quantize_fp_tensor};
use orion::performance::functional::quantize_linear::quantize_linear_u32::quantize_linear;

impl Performance_u32 of PerfomanceTrait<u32> {
    fn quantize_linear(self: @Tensor<u32>) -> Tensor<u32> {
        quantize_tensor(self)
    }

    fn quantize_linear_from_fp(self: @Tensor<FixedType>) -> Tensor<u32> {
        quantize_fp_tensor(self)
    }

    fn quantize_linear_new(
        self: @Tensor<u32>, y_scale: @Tensor<u32>, y_zero_point: @Tensor<u32>
    ) -> Tensor<u32> {
        quantize_linear(self, y_scale, y_zero_point)
    }
}
