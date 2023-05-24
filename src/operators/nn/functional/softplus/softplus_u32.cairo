use core::debug::PrintTrait;
use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{impl_tensor_fp};
use orion::numbers::fixed_point::core::{FixedType,Fixed};
use orion::utils::check_gas;


/// Cf: NNTrait::softplus docstring
fn softplus_u32(z: @Tensor<u32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = Fixed::new_unscaled(1, false);
    loop {
        check_gas();
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = Fixed::new_unscaled(current_index.into(), false);
        let result = (fp_one + fp_current_index.exp()).ln();
        data_result.append(result);
    };
    return TensorTrait::<FixedType>::new(*z.shape, data_result.span());
}

