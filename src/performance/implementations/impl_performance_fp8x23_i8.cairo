use array::ArrayTrait;
use core::option::OptionTrait;

use orion::numbers::signed_integer::i8::i8;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;

impl Performance_fp_i8 of PerfomanceTrait<FP8x23, i8> {
    fn quantize_linear(
        self: @Tensor<FP8x23>, y_scale: @Tensor<FP8x23>, y_zero_point: @Tensor<FP8x23>
    ) -> Tensor<i8> {
        //quantize_linear(self, y_scale, y_zero_point).unwrap()
        panic(array![])
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<FP8x23>, x_zero_point: @Tensor<FP8x23>
    ) -> Tensor<FP8x23> {
        //dequantize_linear(self, x_scale, x_zero_point).unwrap()
        panic(array![])
    }
}
