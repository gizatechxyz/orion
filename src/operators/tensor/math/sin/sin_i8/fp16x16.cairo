use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;


/// Cf: TensorTrait::sin docstring
fn sin(mut self: Tensor<i8>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                result.append(FixedTrait::new_unscaled((*item.mag).into(), *item.sign).sin());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(self.shape, result.span(), self.extra);
}