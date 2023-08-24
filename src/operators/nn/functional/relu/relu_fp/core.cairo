use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::relu::relu_fp::{fp16x16, fp8x23};

/// Cf: NNTrait::relu docstring
fn relu_fp(z: @Tensor<FixedType>) -> Option::<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::relu(*z)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::relu(*z)),
            },
            Option::None(_) => Option::Some((fp16x16::relu(*z))),
        },
        Option::None(_) => Option::Some((fp16x16::relu(*z))),
    }
}

