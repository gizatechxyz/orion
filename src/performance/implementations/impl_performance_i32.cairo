use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_i32::{
    quantize_tensor, quantize_fp8x23_tensor
};

impl Performance_i32_fp8x23 of PerfomanceTrait<i32, fp8x23> {
    fn quantize_linear(self: @Tensor<i32>) -> Tensor<i32> {
        quantize_tensor(self)
    }

    fn quantize_linear_from_fp(self: @Tensor<FixedType<fp8x23>>) -> Tensor<i32> {
        quantize_fp8x23_tensor(self)
    }
}
