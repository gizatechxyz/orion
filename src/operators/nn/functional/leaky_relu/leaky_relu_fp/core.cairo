use orion::operators::tensor::core::{Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::nn::functional::leaky_relu::leaky_relu_fp::{fp16x16, fp8x23};

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu_fp(z: @Tensor<FixedType>, alpha: @FixedType, ) -> Option::<Tensor<FixedType>> {
    match *z.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(fp8x23::leaky_relu(z, alpha)),
                FixedImpl::FP16x16(()) => Option::Some(fp16x16::leaky_relu(z, alpha)),
            },
            Option::None(_) => Option::Some((fp16x16::leaky_relu(z, alpha))),
        },
        Option::None(_) => Option::Some((fp16x16::leaky_relu(z, alpha))),
    }
}

