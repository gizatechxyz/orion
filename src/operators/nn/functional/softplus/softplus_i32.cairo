use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{impl_tensor_fp8x23};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::utils::check_gas;


/// Cf: NNTrait::softplus docstring
fn softplus_i32_fp8x23(z: @Tensor<i32>) -> Tensor<FixedType<fp8x23>> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::<fp8x23>::new_unscaled(1, false);
    loop {
        check_gas();
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index: FixedType = FixedTrait::new_unscaled(
            current_index.mag.into(), current_index.sign
        );
        let result = (fp_one + fp_current_index.exp()).ln();
        data_result.append(result);
    };
    return TensorTrait::new(*z.shape, data_result.span());
}

