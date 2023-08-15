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


/// Cf: NNTrait::softsign docstring
fn softsign(z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::new_unscaled(1, false);
    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let result = current_index / (fp_one + current_index.abs());
        data_result.append(result);
    };
    return TensorTrait::new(*z.shape, data_result.span(), *z.extra);
}

