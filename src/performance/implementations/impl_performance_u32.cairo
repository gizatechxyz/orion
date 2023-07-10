use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantize_linear::quantize_linear_u32::quantize_linear;
use orion::performance::functional::dequantize_linear::dequantize_linear_u32::dequantize_linear;

impl Performance_u32 of PerfomanceTrait<u32, u32> {
    fn quantize_linear(
        self: @Tensor<u32>, y_scale: @Tensor<u32>, y_zero_point: @Tensor<u32>
    ) -> Tensor<u32> {
        quantize_linear(self, y_scale, y_zero_point)
    }

    fn dequantize_linear(
        self: @Tensor<u32>, x_scale: @Tensor<u32>, x_zero_point: @Tensor<u32>
    ) -> Tensor<u32> {
        dequantize_linear(self, x_scale, x_zero_point)
    }
}
