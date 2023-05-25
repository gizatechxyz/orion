use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{impl_tensor_fp};
use orion::numbers::fixed_point::core::{FixedType,Fixed};
use orion::utils::check_gas;


/// Cf: NNTrait::softsign docstring
fn softsign_i32(z: @Tensor<i32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = Fixed::new(1, false);
    loop {
        check_gas();
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = Fixed::new(current_index.mag.into(), current_index.sign);
        let result = fp_current_index / (fp_one + fp_current_index.abs());
        data_result.append(result);
    };
    return TensorTrait::<FixedType>::new(*z.shape, data_result.span());
}

