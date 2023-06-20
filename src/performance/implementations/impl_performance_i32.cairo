use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::core::FixedType;

use orion::operators::tensor::core::Tensor;
use orion::performance::core::PerfomanceTrait;
use orion::performance::functional::quantization::quant_i32::{
    quantize_tensor, quantize_fp_tensor
};

impl Performance_i32 of PerfomanceTrait<i32> {
    fn quantize_linear(self: @Tensor<i32>) -> Tensor<i32> {
        quantize_tensor(self)
    }

    fn quantize_linear_from_fp(self: @Tensor<FixedType>) -> Tensor<i32> {
        quantize_fp_tensor(self)
    }
}
