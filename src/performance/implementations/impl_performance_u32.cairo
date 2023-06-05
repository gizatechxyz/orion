use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_u32::{quantize_tensor, quantize_fp_tensor};

impl Performance_i32_fp8x23 of PerfomanceTrait<u32, fp8x23> {
    fn quantize_linear(self: @Tensor<u32>) -> Tensor<u32> {
        quantize_tensor(self)
    }

    fn quantize_linear_from_fp(self: @Tensor<FixedType>) -> Tensor<u32> {
        quantize_fp_tensor(self)
    }
}
