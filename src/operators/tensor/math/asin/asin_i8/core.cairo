use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{Tensor};
use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::math::asin::asin_i8::fp8x23;
use orion::operators::tensor::math::asin::asin_i8::fp16x16;

/// Cf: TensorTrait::asin docstring
fn asin_i8(self: @Tensor<i8>) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::asin(self)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::asin(self)),
            },
            Option::None(_) => Option::Some(fp16x16::asin(self)),
        },
        Option::None(_) => Option::Some(fp16x16::asin(self)),
    }
}
