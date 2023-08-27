use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23Add, FP8x23Div};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};

/// Cf: NNTrait::sigmoid docstring
fn sigmoid_i32(mut z: Tensor<i32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let current_item = *item * IntegerTrait::new(1, true);
                let fp_current_index = FixedTrait::new_unscaled(
                    current_item.mag.into(), current_item.sign
                );
                let result = FixedTrait::ONE() / (FixedTrait::ONE() + fp_current_index.exp());
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(z.shape, data_result.span(), z.extra);
}

