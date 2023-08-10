use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Add, FP16x16Div};


/// Cf: NNTrait::softplus docstring
fn softplus(z: @Tensor<i8>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::new_unscaled(1, false);
    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index: FixedType = FixedTrait::new_unscaled(
            current_index.mag.into(), current_index.sign
        );
        let result = (fp_one + fp_current_index.exp()).log();
        data_result.append(result);
    };
    return TensorTrait::new(*z.shape, data_result.span(), *z.extra);
}

