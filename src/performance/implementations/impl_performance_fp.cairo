use array::ArrayTrait;
use core::option::OptionTrait;

use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;
use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp_i32::core::quantize_linear as quantize_linear_fp_i32;
use orion::performance::functional::quantize_linear::quantize_linear_fp::fp_u32::core::quantize_linear as quantize_linear_fp_u32;


impl Performance_fp_i32 of PerfomanceTrait<FixedType, i32> {
    fn quantize_linear(
        self: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
    ) -> Tensor<i32> {
        quantize_linear_fp_i32(self, y_scale, y_zero_point).unwrap()
    }

    fn dequantize_linear(
        self: @Tensor<FixedType>, x_scale: @Tensor<FixedType>, x_zero_point: @Tensor<FixedType>
    ) -> Tensor<FixedType> {
        let mut not_supported = ArrayTrait::new();
        not_supported.append('Not supported with FixedType');
        panic(not_supported)
    }
}

impl Performance_fp_u32 of PerfomanceTrait<FixedType, u32> {
    fn quantize_linear(
        self: @Tensor<FixedType>, y_scale: @Tensor<FixedType>, y_zero_point: @Tensor<FixedType>
    ) -> Tensor<u32> {
        quantize_linear_fp_u32(self, y_scale, y_zero_point).unwrap()
    }

    fn dequantize_linear(
        self: @Tensor<FixedType>, x_scale: @Tensor<FixedType>, x_zero_point: @Tensor<FixedType>
    ) -> Tensor<FixedType> {
        let mut not_supported = ArrayTrait::new();
        not_supported.append('Not supported with FixedType');
        panic(not_supported)
    }
}
