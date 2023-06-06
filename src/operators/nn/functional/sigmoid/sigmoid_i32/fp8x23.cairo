use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::utils::check_gas;


/// Cf: NNTrait::sigmoid docstring
fn sigmoid_i32(z: @Tensor<i32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::new_unscaled(1, false);
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap() * IntegerTrait::new(1_u32, true);
        let fp_current_index = FixedTrait::new_unscaled(
            current_index.mag.into(), current_index.sign
        );
        let result = fp_one / (fp_one + fp_current_index.exp());
        data_result.append(result);
    };
    return TensorTrait::<FixedType>::new(*z.shape, data_result.span(), *z.extra);
}

