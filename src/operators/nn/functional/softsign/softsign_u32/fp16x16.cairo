use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Add, FP16x16Div};


/// Cf: NNTrait::softsign docstring
fn softsign(z: @Tensor<u32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::new(1, false);
    loop {
        
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = FixedTrait::new(current_index.into(), false);
        let result = fp_current_index / (fp_one + fp_current_index.abs());
        data_result.append(result);
    };
    return TensorTrait::new(*z.shape, data_result.span(), *z.extra);
}
