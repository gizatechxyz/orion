use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantize_linear::quantize_linear_i32::quantize_linear;

impl Performance_i32 of PerfomanceTrait<i32, i32> {
    fn quantize_linear(
        self: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
    ) -> Tensor<i32> {
        quantize_linear(self, y_scale, y_zero_point)
    }
}
