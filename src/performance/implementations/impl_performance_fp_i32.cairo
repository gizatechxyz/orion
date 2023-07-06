use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_i32::{quantize_tensor, quantize_fp_tensor};
use orion::performance::functional::quantize_linear::quantize_linear_fp::quantize_linear;

impl Performance_fp_i32 of PerfomanceTrait<FixedType, i32> {
    fn quantize_linear(
        self: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
    ) -> Tensor<i32> {
        quantize_linear(self, y_scale, y_zero_point)
    }
}
