use orion::numbers::signed_integer::{i32::i32};
use orion::operators::tensor::core::{Tensor};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::sigmoid::sigmoid_i32::fp8x23;
use orion::operators::nn::functional::sigmoid::sigmoid_i32::fp16x16;

/// Cf: NNTrait::sigmoid docstring
fn sigmoid_i32(z: @Tensor<i32>) -> Option<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::sigmoid_i32(*z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::sigmoid_i32(*z)),
            },
            Option::None(_) => Option::Some(fp16x16::sigmoid_i32(*z)),
        },
        Option::None(_) => Option::Some(fp16x16::sigmoid_i32(*z)),
    }
}

