use core::debug::PrintTrait;
use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16Add, FP16x16Div
};


/// Cf: NNTrait::softplus docstring
fn softplus(mut z: Tensor<u32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let fp_current_item = FixedTrait::new_unscaled((*item).into(), false);
                let result = (FixedTrait::ONE() + fp_current_item.exp()).ln();
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span(), z.extra);
}

