use orion::operators::tensor::core::{Tensor};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::sigmoid::sigmoid_fp::fp8x23;
use orion::operators::nn::functional::sigmoid::sigmoid_fp::fp16x16;

/// Cf: NNTrait::sigmoid docstring
fn sigmoid_fp(z: @Tensor<FixedType>) -> Option<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::sigmoid_fp(*z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::sigmoid_fp(*z)),
            },
            Option::None(_) => Option::Some(fp16x16::sigmoid_fp(*z)),
        },
        Option::None(_) => Option::Some(fp16x16::sigmoid_fp(*z)),
    }
}

