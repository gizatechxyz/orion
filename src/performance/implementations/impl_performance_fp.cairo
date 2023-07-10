use array::ArrayTrait;
use core::option::OptionTrait;

use orion::numbers::signed_integer::i8::i8;
use orion::numbers::fixed_point::core::FixedType;
use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp_i8::core::quantize_linear;
use orion::performance::functional::dequantize_linear::dequantize_linear_fp::core::dequantize_linear;


impl Performance_fp of PerfomanceTrait<FixedType> {
    fn quantize_linear(
        self: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
    ) -> Tensor<i8> {
        quantize_linear(self, y_scale, y_zero_point).unwrap()
    }

    fn dequantize_linear(
        self: @Tensor<i8>, x_scale: @Tensor<FixedType>, x_zero_point: @Tensor<FixedType>
    ) -> Tensor<FixedType> {
        dequantize_linear(self, x_scale, x_zero_point).unwrap()
    }
}
