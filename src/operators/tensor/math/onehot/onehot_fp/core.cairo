use core::option::OptionTrait;
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::onehot::onehot_fp::fp8x23;
use orion::operators::tensor::math::onehot::onehot_fp::fp16x16;
use debug::PrintTrait;

/// Cf: TensorTrait::cumsum docstring
fn onehot(
    self: @Tensor<FixedType>, depth: usize, axis: Option<usize>, values: Span<usize>
) -> Option<Tensor<FixedType>> {
    match *self.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::onehot(self, depth, axis, values)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::onehot(self, depth, axis, values)
                ),
            },
            Option::None(_) => Option::Some(fp16x16::onehot(self, depth, axis, values)),
        },
        Option::None(_) => Option::Some(fp16x16::onehot(self, depth, axis, values)),
    }
}
