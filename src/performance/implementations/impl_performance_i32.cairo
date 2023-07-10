use orion::numbers::signed_integer::{i32::i32, i8::i8};
use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantize_linear::quantize_linear_i32::quantize_linear;
use orion::performance::functional::dequantize_linear::dequantize_linear_i32::dequantize_linear;


impl Performance_i32_i8 of PerfomanceTrait<i32, i8> {
    fn quantize_linear(
        self: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
    ) -> Tensor<i8> {
        quantize_linear(self, y_scale, y_zero_point)
    }

    fn dequantize_linear(
        self: @Tensor<i32>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
    ) -> Tensor<i32> {
        dequantize_linear(self, x_scale, x_zero_point)
    }
}
