use orion::numbers::signed_integer::{i32::i32, i8::i8};

use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::implementations::tensor_i8_fp8x23;
use orion::operators::tensor::implementations::tensor_i32_fp8x23;

use orion::performance::core::PerfomanceTrait;
use orion::performance::functional;


impl Performance_i32_i8 of PerfomanceTrait<i32, i8> {
    fn quantize_linear(
        self: @Tensor<i32>, y_scale: @Tensor<i32>, y_zero_point: @Tensor<i32>
    ) -> Tensor<i8> {
        functional::quantize_linear::quantize_linear(
            self, y_scale, y_zero_point, i32 { mag: 128, sign: true }, i32 { mag: 127, sign: false }
        )
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<i32>, x_zero_point: @Tensor<i32>
    ) -> Tensor<i32> {
        //dequantize_linear(self, x_scale, x_zero_point)
        panic(array![])
    }
}
