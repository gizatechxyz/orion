use core::option::OptionTrait;
use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::concat::concat_fp::fp8x23;
use orion::operators::tensor::math::concat::concat_fp::fp16x16;
use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use core::clone::Clone;
use array::{ArrayTrait, SpanTrait};
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::{impl_tensor_i32, impl_tensor_u32};
use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;

/// Cf: TensorTrait::concat docstring
fn concat(
    mut tensors: Span<Tensor<FixedType>>, axis: usize, 
) -> Option<Tensor<FixedType>> {
    assert(tensors.len() >= 2, 'Wrong values dimensions');

    let mut first = *tensors.at(0);

    match first.extra {
        Option::Some(extra_params) => match extra_params.fixed_point {
            Option::Some(fixed_point) => match fixed_point {
                FixedImpl::FP8x23(()) => Option::Some(
                    fp8x23::concat(tensors, axis)
                ),
                FixedImpl::FP16x16(()) => Option::Some(
                    fp16x16::concat(tensors, axis)
                ),
            },
            Option::None(_) => Option::Some(fp16x16::concat(tensors, axis)),
        },
        Option::None(_) => Option::Some(fp16x16::concat(tensors, axis)),
    }
}
